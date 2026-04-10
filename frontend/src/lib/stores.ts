import { writable } from 'svelte/store';
import type { Corpus, Conversation, Message, Mode, CitationStyle } from './types';

export const corpora = writable<Corpus[]>([]);
export const activeCorpus = writable<Corpus | null>(null);
export const conversations = writable<Conversation[]>([]);
export const activeConversation = writable<Conversation | null>(null);
export const messages = writable<Message[]>([]);
export const streamingContent = writable<string>('');
export const isStreaming = writable<boolean>(false);
export const theme = writable<'light' | 'dark'>(
  typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark' : 'light'
);
