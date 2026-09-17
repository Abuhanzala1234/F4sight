"""Initial schema — the §6 data model.

Built from the ORM metadata rather than hand-transcribed DDL. For an initial
revision that is the safer choice: fifteen tables with twenty-odd CHECK
constraints transcribed by hand is fifteen chances to let the database and the
ORM disagree, and a disagreement there means an invariant that is enforced in
one place and not the other. Later revisions are explicit, as usual.

Requires ``uuid7()`` and the pgvector extension, both created by
``infra/postgres/init.sql`` when the container first starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sqlalchemy as sa
from alembic import op

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ibvap_api.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # Belt and braces: init.sql creates these on a fresh container, but a
    # migration run against an existing database should not fail for want of
    # an extension it can create itself.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    uuid7_exists = bind.execute(
        sa.text("SELECT count(*) FROM pg_proc WHERE proname = 'uuid7'")
    ).scalar_one()
    if not uuid7_exists:
        raise RuntimeError(
            "the uuid7() function is missing. It is created by "
            "infra/postgres/init.sql on first container start. Run "
            "`make nuke && make up`, or apply that file by hand."
        )

    Base.metadata.create_all(bind=bind)

    # Face embeddings: pgvector, opt-in, and reachable only through audited
    # admin enrolment (P6, §7.10). Declared here rather than in the ORM so that
    # the vector type is not a hard import dependency for anyone who never
    # turns faces on.
    op.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_face_vector (
            id          uuid PRIMARY KEY DEFAULT uuid7(),
            person_id   uuid NOT NULL REFERENCES watchlist_person(id) ON DELETE CASCADE,
            embedding   vector(512) NOT NULL,
            source_image_key text,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_face_vector_cosine
        ON watchlist_face_vector USING hnsw (embedding vector_cosine_ops)
        """)

    # P2, enforced by the database as well as by risk.py and its property tests.
    # An alert whose contributions do not add up is an alert whose explanation
    # is a lie, and it must not be storable by any path — including psql.
    op.execute("""
        CREATE OR REPLACE FUNCTION check_risk_breakdown_sums()
        RETURNS trigger AS $$
        DECLARE total numeric;
        BEGIN
            SELECT COALESCE(sum((c->>'weight')::numeric), 0)
              INTO total
              FROM jsonb_array_elements(NEW.risk_breakdown) AS c;
            IF abs(total - NEW.risk_score) > 0.01 THEN
                RAISE EXCEPTION
                  'risk_breakdown sums to % but risk_score is % (alert %). '
                  'The additive risk model is Principle P2; see BUILD_SPEC 7.8.',
                  total, NEW.risk_score, NEW.id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """)
    op.execute("""
        CREATE TRIGGER trg_alert_risk_sums
        BEFORE INSERT OR UPDATE OF risk_score, risk_breakdown ON alert
        FOR EACH ROW EXECUTE FUNCTION check_risk_breakdown_sums()
        """)

    # The audit log is append-only (§12). Blocking UPDATE and DELETE at the
    # database is what makes "an insider edited the record" detectable rather
    # than merely discouraged.
    op.execute("""
        CREATE OR REPLACE FUNCTION audit_log_is_append_only()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only (BUILD_SPEC 12)';
        END;
        $$ LANGUAGE plpgsql
        """)
    op.execute("""
        CREATE TRIGGER trg_audit_append_only
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION audit_log_is_append_only()
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_audit_append_only ON audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit_log_is_append_only()")
    op.execute("DROP TRIGGER IF EXISTS trg_alert_risk_sums ON alert")
    op.execute("DROP FUNCTION IF EXISTS check_risk_breakdown_sums()")
    op.execute("DROP TABLE IF EXISTS watchlist_face_vector")
    Base.metadata.drop_all(bind=op.get_bind())
