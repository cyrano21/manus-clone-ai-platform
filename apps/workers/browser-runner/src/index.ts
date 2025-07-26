import { Worker } from 'bullmq';
import { chromium, Browser, Page } from 'playwright';
import Redis from 'ioredis';
import { BrowserAction, BrowserResult, S3StorageClient } from '@manus/shared';

const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
const storage = S3StorageClient.fromEnv();

class BrowserRunner {
  private browser: Browser | null = null;
  private pages: Map<string, Page> = new Map();

  async initialize() {
    this.browser = await chromium.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    console.log('Browser runner initialized');
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
    this.pages.clear();
  }

  async executeAction(jobId: string, stepId: string, action: BrowserAction): Promise<BrowserResult> {
    if (!this.browser) {
      throw new Error('Browser not initialized');
    }

    const startTime = Date.now();
    let page = this.pages.get(jobId);

    try {
      // Create new page if needed
      if (!page) {
        page = await this.browser.newPage();
        this.pages.set(jobId, page);
      }

      let result: any = null;
      let screenshot: string | undefined;

      switch (action.type) {
        case 'navigate':
          if (!action.url) throw new Error('URL required for navigate action');
          await page.goto(action.url, { waitUntil: 'networkidle' });
          result = { url: page.url() };
          break;

        case 'click':
          if (!action.selector) throw new Error('Selector required for click action');
          await page.click(action.selector);
          result = { clicked: action.selector };
          break;

        case 'type':
          if (!action.selector || !action.value) {
            throw new Error('Selector and value required for type action');
          }
          await page.fill(action.selector, action.value);
          result = { typed: action.value, selector: action.selector };
          break;

        case 'scroll':
          await page.evaluate(() => window.scrollBy(0, window.innerHeight));
          result = { scrolled: true };
          break;

        case 'screenshot':
          const screenshotBuffer = await page.screenshot({ fullPage: true });
          const key = storage.generateKey('screenshot', 'png');
          const uploadResult = await storage.upload(key, screenshotBuffer, 'image/png');
          screenshot = uploadResult.url;
          result = { screenshot: uploadResult.url };
          break;

        case 'extract':
          if (!action.selector) throw new Error('Selector required for extract action');
          const elements = await page.locator(action.selector).all();
          const extractedData = await Promise.all(
            elements.map(async (el) => ({
              text: await el.textContent(),
              html: await el.innerHTML(),
            }))
          );
          result = { extracted: extractedData };
          break;

        default:
          throw new Error(`Unknown action type: ${action.type}`);
      }

      const duration = Date.now() - startTime;

      // Publish event to Redis
      await redis.publish('browser-events', JSON.stringify({
        type: 'step.completed',
        jobId,
        stepId,
        data: { action, result, duration, screenshot },
        timestamp: new Date(),
      }));

      return {
        success: true,
        data: result,
        screenshot,
        duration,
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      // Publish error event
      await redis.publish('browser-events', JSON.stringify({
        type: 'step.failed',
        jobId,
        stepId,
        data: { action, error: errorMessage, duration },
        timestamp: new Date(),
      }));

      return {
        success: false,
        error: errorMessage,
        duration,
      };
    }
  }

  async closePage(jobId: string) {
    const page = this.pages.get(jobId);
    if (page) {
      await page.close();
      this.pages.delete(jobId);
    }
  }
}

const browserRunner = new BrowserRunner();

// Worker to process browser jobs
const worker = new Worker('browser-queue', async (job) => {
  const { jobId, stepId, action } = job.data;
  
  console.log(`Processing browser action: ${action.type} for job ${jobId}`);
  
  const result = await browserRunner.executeAction(jobId, stepId, action);
  
  return result;
}, {
  connection: redis,
  concurrency: 5, // Process up to 5 browser actions concurrently
});

worker.on('completed', (job) => {
  console.log(`Browser job ${job.id} completed`);
});

worker.on('failed', (job, err) => {
  console.error(`Browser job ${job?.id} failed:`, err);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down browser runner...');
  await worker.close();
  await browserRunner.cleanup();
  await redis.disconnect();
  process.exit(0);
});

// Initialize browser on startup
browserRunner.initialize().catch(console.error);

console.log('Browser runner worker started');
