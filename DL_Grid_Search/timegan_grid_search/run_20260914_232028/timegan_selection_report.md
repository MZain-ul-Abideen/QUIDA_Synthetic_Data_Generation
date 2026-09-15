# TimeGAN hyperparameter selection — run 20260914_232028

## Selected configuration: **C03** — hidden = 16, lr = 0.0005, batch = 16, joint iterations = 5000
Composite score **0.6173 ± 0.0014** over 2 seeds (range 0.6163–0.6183); fidelity 0.874, diversity 0.913, novelty 0.065.

**Fidelity.** MMD 0.0894 (best in grid 0.0477), Wasserstein 0.3011, Hellinger 0.2580, correlation difference 0.3850, variance difference 0.326. That places it at the 67% mark of the repeated configurations on fidelity. Fidelity is a genuine strength of this configuration.

**Diversity.** Coverage 1.000, density 1.813. Most of the held-out real manifold is reached by the synthetic set.

**Novelty.** NN ratio 0.873 (synthetic→real nearest-neighbour distance divided by real→real; ≈1 means synthetic sequences sit as far from real ones as real ones sit from each other, ≪1 means copying). Memorisation diagnostic 0.343 — a substantial share of synthetic sequences are near-copies of real ones; treat this configuration as partly a resampler and confirm with the downstream utility check.

**Stability across seeds.** SD 0.0014 (CV 0.2%), min–max spread 0.0020, versus a median CV of 2.2% across the repeated configurations. This is one of the more reproducible configurations — important for an adversarial model that will be re-trained in the next pipeline stage.

**Training behaviour.** The discriminator was updated in 100% of joint iterations (it is updated only when its loss exceeds 0.15). A very high rate means the discriminator is struggling throughout; the generator may be exploiting a weak critic rather than matching the real distribution. Final losses (diagnostic only, never used for ranking): G 21.592, D 1.787, E 1.035.

**Overfitting / memorisation.** MMD against the training split minus MMD against the held-out split is +0.0255. Synthetic data is no closer to the training sequences than to unseen ones, so the generator is not simply replaying what it saw.

**Mode collapse.** Coverage 1.000, density 1.813, variance difference 0.326. No collapse signature: the synthetic set spreads across the real manifold with a comparable variance.

**Does more capacity help?** Mean composite score by hidden dimension: 16 → 0.578, 24 → 0.534, 32 → 0.573. Larger hidden dimensions do not buy quality here, which is consistent with 79 training sequences — the smaller model is preferred.

**Does longer training help?** Mean composite score by joint iterations: 5000 → 0.593, 10000 → 0.551, 20000 → 0.539. Training beyond 5000 iterations does not improve the score, so the extra cost is not justified.

**Is the learning rate stable?** Mean ± SD of the composite score by learning rate: 0.0001 → 0.591 ± 0.016, 0.0005 → 0.572 ± 0.040, 0.001 → 0.520 ± 0.096. The selected 0.0005 is not the least variable setting (0.0001 is); if you see unstable training in the next stage, that is the first parameter to revisit.

**Marginal differences reported honestly.** 2 configurations lie within one standard error of the leader (0.6077): C01 (h=32, it=5000, 0.6215). Those differences are inside seed noise, so the cheaper/simpler configuration was chosen rather than the single highest number.
Concretely: **C01** achieved the highest mean composite score (0.6215, SD 0.0195, hidden 32, 5000 iterations), but **C03** (0.6173, SD 0.0014, hidden 16, 5000 iterations) is statistically indistinguishable from it while being cheaper to train. **C03** is therefore recommended. If you prefer to report the maximiser instead, both are defensible — state the rule you used.

**Caveats to carry forward.** The reference set is 20 held-out real sequences; coverage and density are noisy at that size and the multivariate Hellinger distance saturates. This selection is a *ranking* over configurations, not an absolute quality certificate — the downstream sample-size study and post-evaluation remain the real test.
