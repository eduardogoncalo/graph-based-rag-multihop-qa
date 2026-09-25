CREATE TABLE IF NOT EXISTS datasets (
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    name text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS documents (
    document_id text PRIMARY KEY,
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    title text,
    text text NOT NULL,
    source_path text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES datasets (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents (document_id),
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    method_id text,
    text text NOT NULL,
    start_char integer,
    end_char integer,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES datasets (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS questions (
    question_id text PRIMARY KEY,
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    document_id text REFERENCES documents (document_id),
    question text NOT NULL,
    question_type text NOT NULL,
    clause_type text,
    gold_answer jsonb,
    gold_evidence jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES datasets (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS gold_evidence (
    evidence_id text PRIMARY KEY,
    question_id text NOT NULL REFERENCES questions (question_id),
    document_id text NOT NULL REFERENCES documents (document_id),
    text text NOT NULL,
    start_char integer,
    end_char integer,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id text PRIMARY KEY,
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES datasets (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES experiments (experiment_id),
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    method_id text NOT NULL,
    agent_mode text NOT NULL,
    status text NOT NULL DEFAULT 'created',
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES datasets (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    retrieval_result_id text PRIMARY KEY,
    run_id text REFERENCES runs (run_id),
    dataset_id text NOT NULL,
    dataset_version text NOT NULL,
    method_id text NOT NULL,
    query text NOT NULL,
    latency_ms double precision NOT NULL,
    top_k integer NOT NULL,
    raw_response jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (dataset_id, dataset_version)
        REFERENCES datasets (dataset_id, dataset_version)
);

CREATE TABLE IF NOT EXISTS retrieval_items (
    retrieval_item_id text PRIMARY KEY,
    retrieval_result_id text NOT NULL REFERENCES retrieval_results (retrieval_result_id),
    rank integer NOT NULL,
    item_id text NOT NULL,
    source_document_id text,
    source_chunk_id text,
    score double precision,
    text text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs (run_id),
    question_id text REFERENCES questions (question_id),
    method_id text NOT NULL,
    agent_mode text NOT NULL,
    answer_text text NOT NULL,
    citations jsonb NOT NULL DEFAULT '[]'::jsonb,
    latency_ms double precision,
    prompt_tokens integer,
    completion_tokens integer,
    total_tokens integer,
    estimated_cost double precision,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs (run_id),
    agent_name text NOT NULL,
    role text NOT NULL,
    content text NOT NULL,
    token_count integer,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    evaluation_result_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs (run_id),
    metric_name text NOT NULL,
    metric_value double precision NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
