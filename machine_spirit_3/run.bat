@echo off
set RUST_LOG=info
set HIVEMIND_GATEWAY_URL=http://localhost:6089

echo ═══════════════════════════════════════
echo   Machine Spirit 3
echo   The soul lives with the scripture.
echo ═══════════════════════════════════════
echo.

cargo run --release
