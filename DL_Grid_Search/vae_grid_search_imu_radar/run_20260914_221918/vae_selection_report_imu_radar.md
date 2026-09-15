# β-VAE hyperparameter selection — IMU_RADAR (IMU + Radar) — run 20260914_221918

## Selected configuration: **C04** — β = 0.5, latent dim = 8, lr = 0.0005, batch size = 16
Composite score **0.7895 ± 0.0216** over 3 seeds (fidelity 0.926, diversity 0.516, novelty 0.927).

**Fidelity.** MMD 0.0303 (grid best 0.0200), Wasserstein 0.3061, Hellinger 0.1828, correlation difference 0.4300. This configuration sits in the 88% mark of the repeated configurations on fidelity. Fidelity is a genuine strength here.

**Diversity.** Coverage 1.000 (fraction of held-out real samples with a synthetic neighbour inside their k=5 radius), density 0.748, variance difference 0.614. Coverage is high, so the generator is not collapsing onto a subset of the real manifold.

**Novelty.** NN ratio 1.157 (synthetic→real nearest-neighbour distance divided by the real→real one; 1.0 means synthetic samples are as far from real samples as real samples are from each other, <<1 means copying). The memorisation diagnostic is 0.153 — near-duplication of real samples is not a dominant failure mode here.

**Stability across repetitions.** Score SD 0.0216 (coefficient of variation 2.7%), versus a median of 5.5% across the repeated configurations. This is one of the more reproducible configurations.

**Overfitting and memorisation.** Held-out reconstruction loss 27.092 vs training 10.764 (+151.7%): held-out reconstruction is 152% worse than training reconstruction, which with 1137 features and 79 training samples is a real overfitting signal. The train-vs-held-out MMD gap is -0.0123: synthetic data is 40% closer (in MMD) to the training samples than to the held-out ones — the generator leans on the specific samples it saw, so the downstream evaluation should keep using a held-out reference rather than the full real set.

**Is the latent dimension larger than necessary?** 8.0 of 8 latent units are active (posterior mean variance > 1e-2). Essentially all units carry signal, so 8 is not oversized.

**Does β degrade reconstruction too much?** Reconstruction loss 10.764 vs the best in the repeated set 3.910 (+175.3%). β costs a meaningful amount of reconstruction accuracy; it is retained only because the diversity and novelty gains outweigh it in the composite score.

**Robust, not merely the numerical winner.** 2 configurations are within one standard error of the leader (0.7770): C01 (β=0.1, z=32, 0.7843). The differences between them are inside seed noise, so the simpler configuration was chosen rather than the single highest number.

**Caveats worth carrying forward.** The reference set is 20 held-out real samples; coverage and density are noisy at that size, and the Gaussian Hellinger distance requires covariance shrinkage. The selection is therefore a *ranking* over configurations, not an absolute quality certificate — the downstream sample-size study and post-evaluation remain the real test.
