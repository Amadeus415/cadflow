# Image → VLM-described geometry → CadQuery parametric model, skip noisy mesh entirely

## Stage 1: Send image to a VLM with a structured prompt: "Describe this object as primitive CAD operations (extrude, revolve, fillet, etc.) with dimensions"

## Stage 2: VLM output → validated JSON schema of CAD operations

## Stage 3: JSON → CadQuery/Build123d Python code → STEP/STL export

## CLI: Same agent-friendly interface as above, but the pipeline is image → structured description → parametric CAD — no mesh intermediary

### Bonus: Since the VLM step is just a prompt, agents can iterate ("make it 2mm thicker", "add a mounting hole")