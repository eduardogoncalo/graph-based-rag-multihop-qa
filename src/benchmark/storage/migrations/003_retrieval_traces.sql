CREATE TABLE IF NOT EXISTS retrieval_traces (
    trace_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs (run_id),
    trace_status text NOT NULL,
    trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id)
);
