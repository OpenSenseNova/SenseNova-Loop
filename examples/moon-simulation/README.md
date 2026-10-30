# Moon Simulation: NOVA MISSION 07

## Prompt

Use the [original prompt](prompt.md), supplied by the showcase author.

## Harness

The existing showcase examples document the following client setup:

- Application: SenseTime Raccoon Work (商汤小浣熊) desktop client, version **1.1.4**.
- Harness: **boxagent** (小浣熊客户端).
- Model: **SenseNova-Loop-9B-Preview**.

The newly supplied prompt does not independently specify the client version, selected skills, or complete generation configuration for this case. The setup above is a reference workflow, not a verified execution record for this example.

## Reproduction

1. Open Raccoon Work v1.1.4 and configure SenseNova-Loop-9B-Preview as the model, following the reference workflow above.
2. Enter the prompt linked above; the requested telemetry is simulated.
3. Run the task to generate the interactive webpage and requested local resources.
4. Open the generated `index.html` in a browser and check the interactions and requirements in the prompt.

These are reproduction guidance based on the supplied prompt and existing showcase workflow. Exact reproduction is not guaranteed; no standalone harness implementation or original agent trajectory is included.

## Results

- [Final GIF used in the model README](moon-simulation.gif)

Only the existing showcase preview is included. The generated `index.html` and local resources have not been supplied. The prompt's requested deliverables and browser checks are requirements, not claims that those artifacts or checks are included in this repository.
