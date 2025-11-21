# Integration Tests

Integration tests require a running LangFuse instance and may make actual API calls.

## Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and are skipped by default.

### Setup

1. **Generate secrets:**
   ```bash
   openssl rand -hex 32  # Generate for each secret
   ```

2. **Copy template and update secrets:**
   ```bash
   cp .env.langfuse.example .env.langfuse
   # Edit .env.langfuse and replace all CHANGEME values
   ```

3. **Start LangFuse stack:**
   ```bash
   docker-compose up -d
   ```

4. **Access UI and generate API keys:**
   - Open http://localhost:3000
   - Create account and project
   - Go to Settings → API Keys → Create new key
   - Copy public_key and secret_key to `.env.langfuse`

5. **Run integration tests:**
   ```bash
   pytest -m integration -v
   ```

### Running Specific Integration Tests

```bash
# Run all integration tests
pytest -m integration -v

# Run only LangFuse integration tests
pytest tests/test_langfuse_integration.py -m integration -v

# Run specific test
pytest tests/test_langfuse_integration.py::TestLangFuseIntegration::test_langfuse_health_check_real -v
```

## Integration Test Categories

### LangFuse Integration Tests

File: `tests/test_langfuse_integration.py`

Tests:
- **Health Check**: Verifies LangFuse server is reachable
- **Trace Creation**: Tests callback handler creates traces
- **API Connection**: Validates authentication and API access

**Requirements:**
- Running LangFuse instance (via docker-compose)
- Valid API keys in `.env.langfuse`
- `LANGFUSE_ENABLED=true`

**Skip Conditions:**
- Tests automatically skip if `LANGFUSE_ENABLED=false`
- Tests skip if API keys are not configured

## Cleanup

```bash
# Stop services
docker-compose down

# Optional: Remove volumes (deletes all data)
docker volume rm langfuse_postgres langfuse_clickhouse langfuse_minio
```

## Troubleshooting

### LangFuse Services Won't Start

1. Check secrets are generated:
   ```bash
   grep CHANGEME .env.langfuse
   ```
   Should return nothing (all replaced).

2. Check logs:
   ```bash
   docker-compose logs langfuse-web
   docker-compose logs postgres
   ```

3. Verify ports not in use:
   ```bash
   lsof -i :3000  # LangFuse web
   lsof -i :5432  # PostgreSQL
   ```

### Tests Failing

1. Ensure LangFuse is running:
   ```bash
   docker-compose ps | grep langfuse-web
   ```

2. Verify health endpoint:
   ```bash
   curl http://localhost:3000/api/public/health
   ```

3. Check API keys are valid:
   ```bash
   grep LANGFUSE_PUBLIC_KEY .env.langfuse
   grep LANGFUSE_SECRET_KEY .env.langfuse
   ```

### Integration Tests Skip

Tests will automatically skip if:
- `LANGFUSE_ENABLED=false` is set
- LangFuse is not running
- API keys are not configured

This is expected behavior for graceful degradation.

## Note on Test Coverage

Integration tests complement unit tests:
- **Unit tests**: Fast, mocked, no external dependencies
- **Integration tests**: Slower, real services, validate end-to-end behavior

Run unit tests frequently during development:
```bash
pytest -k "not integration"
```

Run integration tests before deployment:
```bash
pytest -m integration -v
```
