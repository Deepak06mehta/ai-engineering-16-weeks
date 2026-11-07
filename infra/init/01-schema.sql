-- Runs automatically on first `make up`.
-- One table gives you BOTH halves of hybrid retrieval:
--   embedding + HNSW  -> vector search
--   tsv       + GIN   -> BM25-style lexical search
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          BIGSERIAL PRIMARY KEY,
  source      TEXT NOT NULL,
  title       TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now()
);

-- NOTE: VECTOR(768) matches nomic-embed-text. Change the dimension to match your
-- embedding model AND write the choice down — changing it later means re-embedding
-- the entire corpus.
CREATE TABLE IF NOT EXISTS chunks (
  id          BIGSERIAL PRIMARY KEY,
  document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  page        INT,
  chunk_index INT NOT NULL,
  content     TEXT NOT NULL,
  tokens      INT,
  embedding   VECTOR(768),
  tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin        ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_id    ON chunks (document_id);
