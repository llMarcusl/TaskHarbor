import test from 'node:test';
import assert from 'node:assert/strict';
import { getDownloadUrl } from '../src/api/aegra.ts';

test('published result download pins the immutable artifact version', () => {
  const url = getDownloadUrl('thread', 'old-session', '/workspace/report.csv', 'version-one');
  const params = new URL(url, 'http://localhost').searchParams;
  assert.equal(params.get('artifact_id'), 'version-one');
  assert.equal(params.get('session_id'), 'old-session');
  assert.equal(params.get('path'), '/workspace/report.csv');
});

test('legacy results without artifact IDs keep their download URLs', () => {
  assert.equal(getDownloadUrl('thread', 'session', 'a b.csv'),
    '/api/threads/thread/files/download?session_id=session&path=a%20b.csv');
});
