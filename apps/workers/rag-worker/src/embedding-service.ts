import { HfInference } from '@huggingface/inference';

export class EmbeddingService {
  private hf: HfInference;
  private model: string;

  constructor() {
    // Use Hugging Face Inference API with a local model or API
    this.hf = new HfInference(process.env.HF_API_TOKEN);
    this.model = process.env.EMBEDDING_MODEL || 'sentence-transformers/all-MiniLM-L6-v2';
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await this.hf.featureExtraction({
        model: this.model,
        inputs: text,
      });

      // Handle different response formats
      if (Array.isArray(response)) {
        // If response is already a flat array of numbers
        if (typeof response[0] === 'number') {
          return response as number[];
        }
        // If response is a 2D array, take the first row
        if (Array.isArray(response[0])) {
          return response[0] as number[];
        }
      }

      throw new Error('Unexpected embedding response format');
    } catch (error) {
      console.error('Error generating embedding:', error);
      
      // Fallback to a simple hash-based embedding for development
      return this.generateFallbackEmbedding(text);
    }
  }

  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    
    // Process in batches to avoid rate limits
    const batchSize = 10;
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchEmbeddings = await Promise.all(
        batch.map(text => this.generateEmbedding(text))
      );
      embeddings.push(...batchEmbeddings);
      
      // Small delay between batches
      if (i + batchSize < texts.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    return embeddings;
  }

  // Fallback embedding generation for development/testing
  private generateFallbackEmbedding(text: string): number[] {
    const dimension = 384; // Match sentence-transformers dimension
    const embedding = new Array(dimension).fill(0);
    
    // Simple hash-based approach for consistent embeddings
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Generate pseudo-random but deterministic values
    const seed = Math.abs(hash);
    let random = seed;
    
    for (let i = 0; i < dimension; i++) {
      random = (random * 9301 + 49297) % 233280;
      embedding[i] = (random / 233280) * 2 - 1; // Normalize to [-1, 1]
    }
    
    // Normalize the vector
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / magnitude);
  }

  // Calculate cosine similarity between two embeddings
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // Batch similarity calculation
  calculateSimilarities(queryEmbedding: number[], candidateEmbeddings: number[][]): number[] {
    return candidateEmbeddings.map(embedding => 
      this.cosineSimilarity(queryEmbedding, embedding)
    );
  }

  // Get embedding dimension
  getDimension(): number {
    return 384; // sentence-transformers/all-MiniLM-L6-v2 dimension
  }
}
