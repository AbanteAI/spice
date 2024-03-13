phase 0: work out interface
- [ ] simple interface for sync or async, streaming or not llm calls
- [ ] first class support for OpenAI
- [ ] first class support for Anthropic
- [ ] config with model aliases, temperature, key management, default models?
- [ ] timing metrics
- [ ] api error handling, retries
- [ ] rate limiting
- [ ] add types
- [ ] add tests
- [ ] support images
- [ ] support other models through LiteLLM

phase 1: initial mentat integration
- [ ] replace mentat's llm calls with spice

phase 2: absorb remaining mentat features
- [ ] token counting
- [ ] cost tracking
- [ ] transcript logging
- [ ] viewer?
- [ ] prompt loading

phase 3: go beyond mentat features
- [ ] prompt rendering *within* SpiceMessages inputs? i.e. parts of messages can be rendered at call time?
- [ ] rerun transcripts with edited prompts
- [ ] auto-tune prompts / analyze prompt performance
