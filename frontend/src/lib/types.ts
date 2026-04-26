export type Mode = 'exact' | 'general' | 'exhaustive' | 'precis';
export type CitationStyle = 'mla' | 'apa' | 'chicago' | 'harvard' | 'asa' | 'sage';
export type CorpusStatus = 'empty' | 'ingesting' | 'ready' | 'error';

export interface Corpus {
  id: string;
  name: string;
  tags: string;
  collection_name: string;
  status: CorpusStatus;
  document_count: number;
  chunk_count: number;
  created_at: string;
}

export interface Document {
  id: string;
  corpus_id: string;
  filename: string;
  file_type: string | null;
  chunks: number;
  status: string;
  author: string;
  title: string;
  year: number;
  error: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  corpus_id: string;
  title: string | null;
  mode: Mode;
  citation_style: CitationStyle;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata_json: string;
  created_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface IngestionStatus {
  job_id: string;
  total_files: number;
  completed_files: number;
  current_file: string | null;
  current_step: string | null;
  status: 'pending' | 'running' | 'done' | 'failed';
  error: string | null;
}

export interface CitationVerification {
  quote: string;
  status: 'verified' | 'approximate' | 'unverified' | 'too_short';
  source: string | null;
  page: number | null;
  similarity: number;
}

export type StreamEvent =
  | { type: 'content'; data: string }
  | { type: 'warning'; data: string }
  | { type: 'verification'; data: CitationVerification[] }
  | { type: 'done'; data: { message_id: string } }
  | { type: 'error'; data: string };
