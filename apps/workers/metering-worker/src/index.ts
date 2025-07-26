import { Worker } from 'bullmq';
import { PrismaClient } from '@prisma/client';
import Redis from 'ioredis';
import { MetricType } from '@manus/shared';

const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
const prisma = new PrismaClient();

interface UsageEvent {
  userId: string;
  type: MetricType;
  value: number;
  cost?: number;
  metadata?: any;
  timestamp: Date;
}

interface JobMetricEvent {
  jobId: string;
  tokensUsed: number;
  duration: number;
  cost: number;
  browserSteps: number;
}

class MeteringWorker {
  async recordUsage(event: UsageEvent) {
    try {
      await prisma.usageMetric.create({
        data: {
          userId: event.userId,
          type: event.type,
          value: event.value,
          cost: event.cost,
          metadata: event.metadata,
          createdAt: event.timestamp,
        },
      });

      console.log(`Recorded usage: ${event.type} = ${event.value} for user ${event.userId}`);
    } catch (error) {
      console.error('Error recording usage:', error);
      throw error;
    }
  }

  async recordJobMetrics(event: JobMetricEvent) {
    try {
      await prisma.jobMetric.upsert({
        where: { jobId: event.jobId },
        update: {
          tokensUsed: event.tokensUsed,
          duration: event.duration,
          cost: event.cost,
          browserSteps: event.browserSteps,
        },
        create: {
          jobId: event.jobId,
          tokensUsed: event.tokensUsed,
          duration: event.duration,
          cost: event.cost,
          browserSteps: event.browserSteps,
        },
      });

      console.log(`Recorded job metrics for job ${event.jobId}`);
    } catch (error) {
      console.error('Error recording job metrics:', error);
      throw error;
    }
  }

  async aggregateUsage(userId: string, period: 'day' | 'month' = 'month') {
    const now = new Date();
    const startDate = new Date();
    
    if (period === 'day') {
      startDate.setDate(now.getDate() - 1);
    } else {
      startDate.setMonth(now.getMonth() - 1);
    }

    try {
      const metrics = await prisma.usageMetric.groupBy({
        by: ['type'],
        where: {
          userId,
          createdAt: {
            gte: startDate,
            lte: now,
          },
        },
        _sum: {
          value: true,
          cost: true,
        },
      });

      const aggregated = metrics.reduce((acc, metric) => {
        acc[metric.type] = {
          value: metric._sum.value || 0,
          cost: metric._sum.cost || 0,
        };
        return acc;
      }, {} as Record<MetricType, { value: number; cost: number }>);

      return aggregated;
    } catch (error) {
      console.error('Error aggregating usage:', error);
      throw error;
    }
  }

  async checkQuotaExceeded(userId: string): Promise<{
    exceeded: boolean;
    quotas: Record<string, { used: number; limit: number; exceeded: boolean }>;
  }> {
    try {
      // Get user's team and quota
      const user = await prisma.user.findUnique({
        where: { id: userId },
        include: {
          teamMembers: {
            include: {
              team: {
                include: {
                  quota: true,
                },
              },
            },
          },
        },
      });

      if (!user || !user.teamMembers[0]?.team?.quota) {
        return { exceeded: false, quotas: {} };
      }

      const quota = user.teamMembers[0].team.quota;
      const usage = await this.aggregateUsage(userId, 'month');

      const quotas = {
        tokens: {
          used: usage.TOKENS?.value || 0,
          limit: quota.tokensLimit,
          exceeded: (usage.TOKENS?.value || 0) > quota.tokensLimit,
        },
        browserTime: {
          used: usage.BROWSER_TIME?.value || 0,
          limit: quota.browserMinutes * 60 * 1000, // Convert to milliseconds
          exceeded: (usage.BROWSER_TIME?.value || 0) > quota.browserMinutes * 60 * 1000,
        },
        storage: {
          used: usage.STORAGE?.value || 0,
          limit: quota.storageGB * 1024 * 1024 * 1024, // Convert to bytes
          exceeded: (usage.STORAGE?.value || 0) > quota.storageGB * 1024 * 1024 * 1024,
        },
      };

      const exceeded = Object.values(quotas).some(q => q.exceeded);

      return { exceeded, quotas };
    } catch (error) {
      console.error('Error checking quota:', error);
      return { exceeded: false, quotas: {} };
    }
  }

  async calculateCost(type: MetricType, value: number): Promise<number> {
    // Simple cost calculation - can be made more sophisticated
    const rates = {
      TOKENS: 0.001, // $0.001 per 1000 tokens
      BROWSER_TIME: 0.01, // $0.01 per minute
      STORAGE: 0.1, // $0.1 per GB per month
      API_CALLS: 0.001, // $0.001 per call
    };

    const rate = rates[type] || 0;
    
    switch (type) {
      case 'TOKENS':
        return (value / 1000) * rate;
      case 'BROWSER_TIME':
        return (value / (60 * 1000)) * rate; // Convert ms to minutes
      case 'STORAGE':
        return (value / (1024 * 1024 * 1024)) * rate; // Convert bytes to GB
      default:
        return value * rate;
    }
  }

  async processUsageEvent(data: Omit<UsageEvent, 'cost'>) {
    const cost = await this.calculateCost(data.type, data.value);
    
    await this.recordUsage({
      ...data,
      cost,
    });

    // Check if quota is exceeded and publish alert if needed
    const quotaCheck = await this.checkQuotaExceeded(data.userId);
    if (quotaCheck.exceeded) {
      await redis.publish('quota-alerts', JSON.stringify({
        userId: data.userId,
        quotas: quotaCheck.quotas,
        timestamp: new Date(),
      }));
    }
  }
}

const meteringWorker = new MeteringWorker();

// Worker for processing usage events
const usageWorker = new Worker('metering-usage-queue', async (job) => {
  await meteringWorker.processUsageEvent(job.data);
}, {
  connection: redis,
  concurrency: 10,
});

// Worker for processing job metrics
const jobMetricsWorker = new Worker('metering-job-queue', async (job) => {
  await meteringWorker.recordJobMetrics(job.data);
}, {
  connection: redis,
  concurrency: 5,
});

// Worker for quota checks
const quotaWorker = new Worker('metering-quota-queue', async (job) => {
  const { userId } = job.data;
  return await meteringWorker.checkQuotaExceeded(userId);
}, {
  connection: redis,
  concurrency: 10,
});

// Scheduled job for daily aggregation
const aggregationWorker = new Worker('metering-aggregation-queue', async (job) => {
  const { userId, period } = job.data;
  return await meteringWorker.aggregateUsage(userId, period);
}, {
  connection: redis,
  concurrency: 3,
});

// Event handlers
[usageWorker, jobMetricsWorker, quotaWorker, aggregationWorker].forEach(worker => {
  worker.on('completed', (job) => {
    console.log(`Metering job ${job.id} completed`);
  });

  worker.on('failed', (job, err) => {
    console.error(`Metering job ${job?.id} failed:`, err);
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down metering workers...');
  await Promise.all([
    usageWorker.close(),
    jobMetricsWorker.close(),
    quotaWorker.close(),
    aggregationWorker.close(),
  ]);
  await prisma.$disconnect();
  await redis.disconnect();
  process.exit(0);
});

console.log('Metering workers started');
