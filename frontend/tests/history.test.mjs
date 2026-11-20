import test from 'node:test';
import assert from 'node:assert/strict';
import { mergeMessages, getConversationMessages } from '../src/api/aegra.ts';

test('compaction values retain history and hide synthetic summary', () => {
  const old = [{ id: '1', type: 'human', content: 'original' },
    { id: '2', type: 'tool', content: 'full result' }];
  const next = [{ id: 's', type: 'ai', content: 'summary', additional_kwargs: { memory_internal: true } },
    { id: '2', type: 'tool', content: 'preview', additional_kwargs: { context_trimmed: true } },
    { id: '3', type: 'ai', content: 'continued' }];
  assert.deepEqual(mergeMessages(old, next), [...old, next[2]]);
});

test('optimistic user ID merges with server echo', () => {
  const user = { id: 'user', type: 'human', content: '/compact decisions' };
  assert.equal(mergeMessages([user], [user]).length, 1);
});

test('reload reads all archive pages before merging working state', async () => {
  const originalFetch = globalThis.fetch;
  const paths = [];
  globalThis.fetch = async (path) => {
    paths.push(path);
    const data = path.endsWith('/state')
      ? { values: { messages: [{ id: 's', type: 'ai', content: 'summary',
          additional_kwargs: { memory_internal: true } },
        { id: '3', type: 'ai', content: 'latest' }] } }
      : path.endsWith('after_seq=0')
        ? { enabled: true, messages: [{ id: '1', type: 'human', content: 'old' }], next_seq: 1 }
        : { enabled: true, messages: [{ id: '2', type: 'ai', content: 'answer' }], next_seq: null };
    return { ok: true, json: async () => data };
  };
  try {
    assert.deepEqual((await getConversationMessages('t')).map(m => m.id), ['1', '2', '3']);
    assert.equal(paths.length, 3);
  } finally { globalThis.fetch = originalFetch; }
});
