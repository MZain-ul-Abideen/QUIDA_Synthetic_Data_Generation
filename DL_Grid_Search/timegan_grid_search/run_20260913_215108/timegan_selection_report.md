# TimeGAN hyperparameter selection — run 20260913_215108

## Selected configuration: **C01** — hidden = 24, lr = 0.0005, batch = 16, joint iterations = 5000
Composite score **0.6499 ± 0.0438** over 2 seeds (range 0.6190–0.6809); fidelity 0.659, diversity 0.616, novelty 0.675.

**Fidelity.** MMD 0.0579 (best in grid 0.0412), Wasserstein 0.2939, Hellinger 0.2092, correlation difference 0.4283, variance difference 0.604. That places it at the 67% mark of the repeated configurations on fidelity. Fidelity is a genuine strength of this configuration.

**Diversity.** Coverage 1.000, density 1.510. Most of the held-out real manifold is reached by the synthetic set.

**Novelty.** NN ratio 1.005 (synthetic→real nearest-neighbour distance divided by real→real; ≈1 means synthetic sequences sit as far from real ones as real ones sit from each other, ≪1 means copying). Memorisation diagnostic 0.278 — a substantial share of synthetic sequences are near-copies of real ones; treat this configuration as partly a resampler and confirm with the downstream utility check.

**Stability across seeds.** SD 0.0438 (CV 6.7%), min–max spread 0.0619, versus a median CV of 6.7% across the repeated configurations. This is one of the more reproducible configurations — important for an adversarial model that will be re-trained in the next pipeline stage.

**Training behaviour.** The discriminator was updated in 100% of joint iterations (it is updated only when its loss exceeds 0.15). A very high rate means the discriminator is struggling throughout; the generator may be exploiting a weak critic rather than matching the real distribution. Final losses (diagnostic only, never used for ranking): G 21.573, D 1.737, E 0.876.

**Overfitting / memorisation.** MMD against the training split minus MMD against the held-out split is +0.0186. Synthetic data is no closer to the training sequences than to unseen ones, so the generator is not simply replaying what it saw.

**Mode collapse.** Coverage 1.000, density 1.510, variance difference 0.604. These values carry a collapse signature (low coverage and/or badly mismatched variance) — the generator concentrates on a narrow region. Inspect Visualization 5 before using this configuration.

**Does more capacity help?** Mean composite score by hidden dimension: 16 → 0.488, 24 → 0.529. The best mean is at hidden = 24, so capacity does contribute, though the spread between settings (0.041) should be weighed against the seed noise in Table 4.

**Does longer training help?** Mean composite score by joint iterations: 5000 → 0.561, 10000 → 0.456. Training beyond 5000 iterations does not improve the score, so the extra cost is not justified.

**Is the learning rate stable?** Mean ± SD of the composite score by learning rate: 0.0001 → 0.507 ± 0.071, 0.0005 → 0.510 ± 0.122. The selected 0.0005 is not the least variable setting (0.0001 is); if you see unstable training in the next stage, that is the first parameter to revisit.

**Marginal differences.** No other configuration reaches the 1-SE threshold (0.6190), so the leader wins outright rather than by a margin inside seed noise.

**Caveats to carry forward.** The reference set is 20 held-out real sequences; coverage and density are noisy at that size and the multivariate Hellinger distance saturates. This selection is a *ranking* over configurations, not an absolute quality certificate — the downstream sample-size study and post-evaluation remain the real test.
