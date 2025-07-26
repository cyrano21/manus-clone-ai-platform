import { Worker } from 'bullmq';
import { HfInference } from '@huggingface/inference';
import Redis from 'ioredis';
import { createVectorClientFromEnv } from '@manus/shared';
import { DocumentProcessor } from './document-processor';
import { EmbeddingService } from './embedding-service';

const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
const vectorClient = createVectorClientFromEnv();
const embeddingService = new EmbeddingService();
const documentProcessor = new DocumentProcessor();

class RAGWorker {
  async processDocument(data: {
    documentId: string;
    content: string;
    contentType: string;
    projectId?: string;
  }) {
    const { documentId, content, contentType, projectId } = data;
    
    console.log(`Processing document ${documentId} of type ${contentType}`);

    try {
      // Extract text from document
      const text = await documentProcessor.extractText(content, contentType);
      
      // Chunk the text
      const chunks = await documentProcessor.chunkText(text, {
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      // Generate embeddings for chunks
      const embeddings = await embeddingService.generateEmbeddings(
        chunks.map(chunk => chunk.content)
      );

      // Prepare vector documents
      const vectorDocs = chunks.map((chunk, index) => ({
        id: `${documentId}-chunk-${index}`,
        content: chunk.content,
        embedding: embeddings[index],
        metadata: {
          documentId,
          projectId,
          chunkIndex: index,
          startOffset: chunk.startOffset,
          endOffset: chunk.endOffset,
        },
      }));

      // Store in vector database
      const collectionName = projectId ? `project-${projectId}` : 'global';
      await vectorClient.createCollection(collectionName, 384); // Sentence transformer dimension
      await vectorClient.upsert(collectionName, vectorDocs);

      // Publish completion event
      await redis.publish('rag-events', JSON.stringify({
        type: 'document.processed',
        documentId,
        chunksCount: chunks.length,
        timestamp: new Date(),
      }));

      console.log(`Successfully processed document ${documentId} with ${chunks.length} chunks`);
      
      return {
        success: true,
        chunksCount: chunks.length,
        collectionName,
      };

    } catch (error) {
      console.error(`Error processing document ${documentId}:`, error);
      
      // Publish error event
      await redis.publish('rag-events', JSON.stringify({
        type: 'document.failed',
        documentId,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date(),
      }));

      throw error;
    }
  }

  async searchDocuments(data: {
    query: string;
    projectId?: string;
    limit?: number;
    threshold?: number;
  }) {
    const { query, projectId, limit = 5, threshold = 0.7 } = data;
    
    console.log(`Searching for: "${query}" in project ${projectId || 'global'}`);

    try {
      // Generate query embedding
      const queryEmbedding = await embeddingService.generateEmbedding(query);
      
      // Search in vector database
      const collectionName = projectId ? `project-${projectId}` : 'global';
      const results = await vectorClient.search(
        collectionName,
        queryEmbedding,
        limit,
        threshold
      );

      console.log(`Found ${results.length} relevant chunks`);

      return {
        success: true,
        results: results.map(result => ({
          id: result.id,
          content: result.content,
          score: result.score,
          documentId: result.metadata.documentId,
          chunkIndex: result.metadata.chunkIndex,
        })),
        query,
      };

    } catch (error) {
      console.error(`Error searching documents:`, error);
      throw error;
    }
  }

  async deleteDocument(data: { documentId: string; projectId?: string }) {
    const { documentId, projectId } = data;
    
    console.log(`Deleting document ${documentId} from project ${projectId || 'global'}`);

    try {
      const collectionName = projectId ? `project-${projectId}` : 'global';
      
      // Find all chunk IDs for this document
      const searchResults = await vectorClient.search(
        collectionName,
        new Array(384).fill(0), // Dummy embedding
        1000, // Large limit to get all chunks
        0 // No threshold
      );
      
      const chunkIds = searchResults
        .filter(result => result.metadata.documentId === documentId)
        .map(result => result.id);

      if (chunkIds.length > 0) {
        await vectorClient.delete(collectionName, chunkIds);
      }

      console.log(`Deleted ${chunkIds.length} chunks for document ${documentId}`);

      return {
        success: true,
        deletedChunks: chunkIds.length,
      };

    } catch (error) {
      console.error(`Error deleting document ${documentId}:`, error);
      throw error;
    }
  }
}

const ragWorker = new RAGWorker();

// Worker for document processing
const processWorker = new Worker('rag-process-queue', async (job) => {
  return await ragWorker.processDocument(job.data);
}, {
  connection: redis,
  concurrency: 3,
});

// Worker for document search
const searchWorker = new Worker('rag-search-queue', async (job) => {
  return await ragWorker.searchDocuments(job.data);
}, {
  connection: redis,
  concurrency: 10,
});

// Worker for document deletion
const deleteWorker = new Worker('rag-delete-queue', async (job) => {
  return await ragWorker.deleteDocument(job.data);
}, {
  connection: redis,
  concurrency: 5,
});

// Event handlers
[processWorker, searchWorker, deleteWorker].forEach(worker => {
  worker.on('completed', (job) => {
    console.log(`RAG job ${job.id} completed`);
  });

  worker.on('failed', (job, err) => {
    console.error(`RAG job ${job?.id} failed:`, err);
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down RAG workers...');
  await Promise.all([
    processWorker.close(),
    searchWorker.close(),
    deleteWorker.close(),
  ]);
  await redis.disconnect();
  process.exit(0);
});

console.log('RAG workers started');
