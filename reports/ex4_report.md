## Dataset & Setup

- Dataset: CIFAR-10 subset (cat, dog, horse), vooraf geëxporteerd naar PNG (ImageFolder).
- Preprocessing: ToTensor (geen data-augmentatie).
- Hardware: CPU-only (WSL2), beperkte parallelisatie (max 2–4 Ray trials).

## Hypotheses
- 1 hoofdhypothese over capaciteit (diepte×filters)
- 1 subhypothese over regularisatie pas nuttig bij grotere modellen
- 1 subhypothese over skip connections → minder LR-gevoelig