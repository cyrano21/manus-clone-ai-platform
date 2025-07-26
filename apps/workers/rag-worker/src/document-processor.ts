import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import csv from 'csv-parser';
import { Readable } from 'stream';

export interface TextChunk {
  content: string;
  startOffset: number;
  endOffset: number;
}

export interface ChunkingOptions {
  chunkSize: number;
  chunkOverlap: number;
  separators?: string[];
}

export class DocumentProcessor {
  async extractText(content: string, contentType: string): Promise<string> {
    switch (contentType) {
      case 'application/pdf':
        return this.extractFromPDF(content);
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return this.extractFromDocx(content);
      case 'text/csv':
        return this.extractFromCSV(content);
      case 'text/markdown':
      case 'text/plain':
        return content;
      default:
        // Try to extract as plain text
        return content;
    }
  }

  private async extractFromPDF(base64Content: string): Promise<string> {
    try {
      const buffer = Buffer.from(base64Content, 'base64');
      const data = await pdfParse(buffer);
      return data.text;
    } catch (error) {
      console.error('Error extracting PDF text:', error);
      throw new Error('Failed to extract text from PDF');
    }
  }

  private async extractFromDocx(base64Content: string): Promise<string> {
    try {
      const buffer = Buffer.from(base64Content, 'base64');
      const result = await mammoth.extractRawText({ buffer });
      return result.value;
    } catch (error) {
      console.error('Error extracting DOCX text:', error);
      throw new Error('Failed to extract text from DOCX');
    }
  }

  private async extractFromCSV(csvContent: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const rows: any[] = [];
      const stream = Readable.from([csvContent]);
      
      stream
        .pipe(csv())
        .on('data', (row) => rows.push(row))
        .on('end', () => {
          // Convert CSV rows to readable text
          const text = rows.map(row => 
            Object.entries(row)
              .map(([key, value]) => `${key}: ${value}`)
              .join(', ')
          ).join('\n');
          resolve(text);
        })
        .on('error', reject);
    });
  }

  async chunkText(text: string, options: ChunkingOptions): Promise<TextChunk[]> {
    const { chunkSize, chunkOverlap, separators = ['\n\n', '\n', '. ', ' '] } = options;
    
    if (text.length <= chunkSize) {
      return [{
        content: text,
        startOffset: 0,
        endOffset: text.length,
      }];
    }

    const chunks: TextChunk[] = [];
    let startOffset = 0;

    while (startOffset < text.length) {
      let endOffset = Math.min(startOffset + chunkSize, text.length);
      
      // Try to find a good breaking point
      if (endOffset < text.length) {
        let bestBreakPoint = endOffset;
        
        for (const separator of separators) {
          const lastIndex = text.lastIndexOf(separator, endOffset);
          if (lastIndex > startOffset) {
            bestBreakPoint = lastIndex + separator.length;
            break;
          }
        }
        
        endOffset = bestBreakPoint;
      }

      const content = text.slice(startOffset, endOffset).trim();
      
      if (content.length > 0) {
        chunks.push({
          content,
          startOffset,
          endOffset,
        });
      }

      // Move start position with overlap
      startOffset = Math.max(startOffset + 1, endOffset - chunkOverlap);
    }

    return chunks;
  }

  // Extract metadata from different document types
  extractMetadata(content: string, contentType: string): Record<string, any> {
    const metadata: Record<string, any> = {
      contentType,
      length: content.length,
      extractedAt: new Date().toISOString(),
    };

    // Add content-specific metadata
    switch (contentType) {
      case 'text/markdown':
        metadata.headings = this.extractMarkdownHeadings(content);
        break;
      case 'text/csv':
        metadata.estimatedRows = content.split('\n').length - 1;
        break;
    }

    return metadata;
  }

  private extractMarkdownHeadings(content: string): string[] {
    const headingRegex = /^(#{1,6})\s+(.+)$/gm;
    const headings: string[] = [];
    let match;

    while ((match = headingRegex.exec(content)) !== null) {
      headings.push(match[2]);
    }

    return headings;
  }

  // Clean and normalize text
  cleanText(text: string): string {
    return text
      // Remove excessive whitespace
      .replace(/\s+/g, ' ')
      // Remove special characters that might interfere with embeddings
      .replace(/[^\w\s\-.,!?;:()\[\]{}'"]/g, '')
      // Trim
      .trim();
  }

  // Split text by sentences for better semantic chunking
  splitBySentences(text: string): string[] {
    // Simple sentence splitting - could be improved with NLP libraries
    return text
      .split(/[.!?]+/)
      .map(sentence => sentence.trim())
      .filter(sentence => sentence.length > 0);
  }
}
