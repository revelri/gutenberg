import type { Corpus, Conversation, ConversationDetail, Document, IngestionStatus, Mode, CitationStyle, StreamEvent } from './types';

const BASE = '/api';

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

// ── Corpus ──────────────────────────────────────────────────────────

export async function listCorpora(): Promise<Corpus[]> {
  return json('/corpus');
}

export async function getCorpus(id: string): Promise<Corpus & { documents: Document[] }> {
  return json(`/corpus/${id}`);
}

export async function createCorpus(name: string, tags: string): Promise<{ id: string }> {
  const form = new FormData();
  form.append('name', name);
  form.append('tags', tags);
  return json('/corpus', { method: 'POST', body: form });
}

export async function deleteCorpus(id: string): Promise<void> {
  await json(`/corpus/${id}`, { method: 'DELETE' });
}

export async function uploadDocuments(corpusId: string, files: File[]): Promise<{ job_id: string }> {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  return json(`/corpus/${corpusId}/ingest`, { method: 'POST', body: form });
}

export async function deleteDocument(corpusId: string, docId: string): Promise<void> {
  await json(`/corpus/${corpusId}/documents/${docId}`, { method: 'DELETE' });
}

export async function reingestDocument(corpusId: string, docId: string): Promise<void> {
  await json(`/corpus/${corpusId}/documents/${docId}/reingest`, { method: 'POST' });
}

export function streamIngestionStatus(corpusId: string, onProgress: (s: IngestionStatus) => void): EventSource {
  const es = new EventSource(`${BASE}/corpus/${corpusId}/ingest/status`);
  es.addEventListener('progress', (e) => onProgress(JSON.parse(e.data)));
  es.addEventListener('error', () => es.close());
  return es;
}

// ── Conversations ───────────────────────────────────────────────────

export async function listConversations(corpusId: string): Promise<Conversation[]> {
  return json(`/corpus/${corpusId}/conversations`);
}

export async function createConversation(
  corpusId: string,
  mode: Mode = 'general',
  citationStyle: CitationStyle = 'chicago'
): Promise<Conversation> {
  return json(`/corpus/${corpusId}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, citation_style: citationStyle }),
  });
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  return json(`/conversations/${id}`);
}

export async function deleteConversation(id: string): Promise<void> {
  await json(`/conversations/${id}`, { method: 'DELETE' });
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await json(`/conversations/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
}

// ── Chat (SSE streaming) ────────────────────────────────────────────

export async function* streamMessage(
  conversationId: string,
  content: string,
  term?: string,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, term }),
    signal,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  if (!res.body) return;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
        continue;
      }
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          switch (currentEvent) {
            case 'message':
              if (data.content) yield { type: 'content', data: data.content };
              break;
            case 'warning':
              yield { type: 'warning', data: data.message || data.error || 'Unknown warning' };
              break;
            case 'verification':
              yield { type: 'verification', data: data.citations || [] };
              break;
            case 'done':
              yield { type: 'done', data: { message_id: data.message_id } };
              break;
            case 'error':
              yield { type: 'error', data: data.error || 'Unknown error' };
              break;
          }
        } catch { /* skip malformed JSON lines */ }
        currentEvent = 'message'; // reset after data line
      }
    }
  }
}
