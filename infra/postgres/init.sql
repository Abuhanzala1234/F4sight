-- IBVAP database bootstrap. Runs once, on first container start.
-- Schema itself lives in Alembic migrations (§6); this file is only for
-- things that must exist before the first migration runs.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_bytes, digest
CREATE EXTENSION IF NOT EXISTS "vector";     -- pgvector, for face embeddings (opt-in)

-- UUIDv7: time-sortable primary keys. Evidence ordering matters (§6), and a
-- random UUID makes "what happened next" an index-less sort.
-- Implementation follows RFC 9562 §5.7: 48-bit big-endian ms timestamp,
-- version 7, variant 0b10, 74 bits of randomness.
CREATE OR REPLACE FUNCTION uuid7() RETURNS uuid AS $$
DECLARE
    unix_ts_ms  bytea;
    uuid_bytes  bytea;
BEGIN
    unix_ts_ms := substring(int8send((extract(epoch FROM clock_timestamp()) * 1000)::bigint) FROM 3);
    uuid_bytes := unix_ts_ms || gen_random_bytes(10);
    -- version 7
    uuid_bytes := set_byte(uuid_bytes, 6, (b'0111' || get_byte(uuid_bytes, 6)::bit(8) << 4 >> 4)::bit(8)::int);
    -- variant 0b10
    uuid_bytes := set_byte(uuid_bytes, 8, (b'10'   || get_byte(uuid_bytes, 8)::bit(8) << 2 >> 2)::bit(8)::int);
    RETURN encode(uuid_bytes, 'hex')::uuid;
END
$$ LANGUAGE plpgsql VOLATILE;

COMMENT ON FUNCTION uuid7() IS
  'RFC 9562 UUIDv7. Time-sortable. Default PK for every IBVAP table.';

-- Every timestamp in this system is UTC. The dashboard localises; the database
-- never guesses.
ALTER DATABASE ibvap SET timezone TO 'UTC';
