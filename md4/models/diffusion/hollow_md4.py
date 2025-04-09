# Copied and edited from md4.py by Yixiu

"""A version of the Simplified masked diffusion (MD4) using hollow transformers."""

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from md4 import binary_search
from md4 import utils
from md4.models import backward

import jax.random as jr
from jax.scipy.special import logsumexp

tfd = tfp.distributions


class MaskingSchedule(nn.Module):
    """Masking noise schedule."""

    data_shape: tuple[int, ...]
    schedule_fn_type: str = "cosine"
    eps: float = 1e-4

    def __call__(self, t):
        # return logSNR
        return jnp.log(self.alpha(t) / (1.0 - self.alpha(t)))

    def _dalpha(self, t):
        if self.schedule_fn_type == "cosine":
            return -math.pi / 2.0 * jax.lax.sin(math.pi / 2.0 * (1.0 - t))
        elif self.schedule_fn_type == "linear":
            return -jnp.ones_like(t)
        elif "poly" in self.schedule_fn_type:
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return -exponent * t ** (exponent - 1.0)
        else:
            raise NotImplementedError()

    def dalpha(self, t):
        return (1.0 - 2 * self.eps) * self._dalpha(t)

    def _alpha(self, t):
        if self.schedule_fn_type == "linear":
            return 1.0 - t
        elif "poly" in self.schedule_fn_type:
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return 1.0 - t**exponent
        elif self.schedule_fn_type == "cosine":
            return 1.0 - jax.lax.cos(math.pi / 2.0 * (1.0 - t))
        else:
            raise NotImplementedError()

    def alpha(self, t):
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps

    def dgamma_times_alpha(self, t):
        return self.dalpha(t) / (1.0 - self.alpha(t))


# TODO: we should really inherit from MD4
class HollowMD4(nn.Module):
    """Simplified masked discrete diffusion model with Hollow Transformers."""

    data_shape: tuple[int, ...]
    cont_time: bool = False
    timesteps: int = 1000
    feature_dim: int = 128

    hidden_dim: int | None = None

    num_heads: int = 12
    antithetic_time_sampling: bool = True
    n_layers: int = 32
    n_layers_per_mixed: int = 8
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    vocab_size: int = 256
    noise_schedule_type: str = "linear"
    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    outside_embed: bool = False
    # time_features: t or none
    time_features: str = "t"
    classes: int = 10 + 1  # image classes
    sampler: str = "analytic"
    # uniform, cosine
    sampling_grid: str = "cosine"
    topp: float = 0.98
    model_sharding: bool = False
    loss_type: str = "mixed"
    
    # Informed correctors
    k: int = 1
    gibbs_temp: float = 1.0
    # Uninformed correctors
    uninformed_step_size: float = 0.5
    # MaskGIT
    maskgit_temp: float = 1.0

    def setup(self):
        self.noise_schedule = MaskingSchedule(self.data_shape, self.noise_schedule_type)

        if self.classes > 0:
            self.cond_embeddings = nn.Embed(self.classes, self.feature_dim)
        self.classifier = backward.DiscreteClassifier(
            n_layers=self.n_layers,
            n_dit_layers=self.n_dit_layers,
            dit_num_heads=self.dit_num_heads,
            dit_hidden_size=self.dit_hidden_size,
            ch_mult=self.ch_mult,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
            use_attn_dropout=self.use_attn_dropout,
            mlp_type=self.mlp_type,
            depth_scaled_init=self.depth_scaled_init,
            cond_type=self.cond_type,
            outside_embed=self.outside_embed,
            model_sharding=self.model_sharding,
            n_layers_per_mixed=self.n_layers_per_mixed,
            use_hollow_transformer=True,
        )

    def forward_sample(self, x, t):
        t = utils.reverse_broadcast(t, x.ndim)
        a = self.noise_schedule.alpha(t)
        un_mask = jax.random.bernoulli(self.make_rng("sample"), a, x.shape)
        # MASK = vocab_size
        return jnp.where(un_mask, x, self.vocab_size)

    def prior_sample(self, batch_size):
        return self.vocab_size * jnp.ones(
            [batch_size] + list(self.data_shape), dtype="int32"
        )

    def get_cond_embedding(self, conditioning):
        if conditioning is not None:
            return self.cond_embeddings(conditioning)
        return None

    def predict_x(self, zt, t, cond=None, train=False):
        t = None if self.time_features == "none" else t
        return self.classifier(zt, t=t, cond=cond, train=train)

    def visualize_classifier(self, x, t, conditioning=None):
        # if it's image, x: [bs, h, w, c]
        # if it's text, x: [bs, seq_len]
        cond = self.get_cond_embedding(conditioning)
        # t: []
        # if it's image, zt: [bs, h, w, c]
        # if it's text, zt: [bs, seq_len]
        zt = self.forward_sample(x, t)
        # logits: [bs, h, w, c, vocab_size] for images
        # [bs, seq_len, vocab_size] for text
        logits, _ = self.predict_x(zt, t, cond=cond)
        n_indep_axes = logits.ndim - 2
        dist = tfd.Independent(tfd.Categorical(logits=logits), n_indep_axes)
        return dist

    def encode(self, x, conditioning=None):
        del conditioning
        return x

    def decode(self, z0, conditioning=None):
        # Remove any mask tokens left in the last step of sampling.
        masked = z0 == self.vocab_size
        z0_cliped = jnp.where(masked, jnp.zeros_like(z0), z0)
        masked = masked[..., None]
        cond = self.get_cond_embedding(conditioning)
        logits, _ = self.predict_x(z0, jnp.array(0.0), cond=cond)
        probs = jnp.where(
            masked,
            nn.softmax(logits, axis=-1),
            jax.nn.one_hot(z0_cliped, self.vocab_size),
        )
        n_indep_axes = probs.ndim - 2
        dist = tfd.Independent(tfd.Categorical(probs=probs), n_indep_axes)
        return dist.mode().astype("int32")

    def recon_loss(self):
        """The reconstruction loss measures the gap in the first step."""
        alpha_t1 = self.noise_schedule.alpha(0.0)
        loss_recon = (
            jnp.prod(jnp.array(self.data_shape))
            * (1.0 - alpha_t1)
            * jnp.log(self.vocab_size)
        )
        return loss_recon

    def latent_loss(self):
        # negligible
        return jnp.array(0.0)

    def diffusion_loss(self, t, x, cond=None, train=False):
        if not self.cont_time:
            # discretize time steps
            t = (jnp.floor(t * self.timesteps) + 1) / self.timesteps

        # sample z_t
        zt = self.forward_sample(x, t)
        logits, _ = self.predict_x(zt, t, cond=cond, train=train)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        one_hot_x = jax.nn.one_hot(x, self.vocab_size)
        neg_cross_ent = one_hot_x * log_p
        neg_cross_ent = jnp.where(one_hot_x, neg_cross_ent, 0.0)
        neg_cross_ent = jnp.sum(neg_cross_ent, axis=-1)
        mask = (zt == self.vocab_size).astype("float32")

        remaining_axis = list(range(x.ndim)[1:])
        # masked_neg_cross_ent: [bs]
        masked_neg_cross_ent = jnp.sum(mask * neg_cross_ent, remaining_axis)
        nonmask_neg_cross_ent = jnp.sum((1 - mask) * neg_cross_ent, remaining_axis)

        if not self.cont_time:
            assert NotImplementedError
            # loss for finite depth T, i.e. discrete time
            s = t - (1.0 / self.timesteps)
            gt = self.noise_schedule(t)
            gs = self.noise_schedule(s)
            loss_diff = (
                self.timesteps
                * jnp.expm1(gt - gs)
                * self.noise_schedule.alpha(s)
                * masked_neg_cross_ent
            )
        else:
            # Efficients of the non-masked positions
            # Alpha does not go to 0 so no worries about divide by 0
            nonmask_coeff = self.noise_schedule.dalpha(t) / self.noise_schedule.alpha(t)

            # cont-time loss
            # Average over the two losses
            loss_diff_mixed = (
                self.noise_schedule.dgamma_times_alpha(t) * masked_neg_cross_ent
                + nonmask_coeff * nonmask_neg_cross_ent
            ) * 0.5
            loss_diff_masked = (
                self.noise_schedule.dgamma_times_alpha(t) * masked_neg_cross_ent
            )
            if self.loss_type == "mixed":
                return loss_diff_mixed, loss_diff_mixed, loss_diff_masked
            elif self.loss_type == "masked":
                return loss_diff_masked, loss_diff_mixed, loss_diff_masked
            else:
                raise NotImplementedError

        # loss_diff: [bs]
        # return loss_diff

    @nn.compact
    def __call__(self, x, cond=None, train=False):
        bs = x.shape[0]
        cond = self.get_cond_embedding(cond)

        # 1. RECONSTRUCTION LOSS: []
        # add noise and reconstruct
        loss_recon = self.recon_loss()

        # 2. LATENT LOSS: []
        loss_prior = self.latent_loss()

        # 3. DIFFUSION LOSS: [bs]
        # sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / bs), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=[bs])

        loss_diff, loss_mixed, loss_masked = self.diffusion_loss(
            t, x, cond=cond, train=train
        )
        loss = loss_diff.mean() + loss_prior + loss_recon

        model_stats = {
            "loss": loss,
            "loss_diff": loss_diff,
            "loss_prior": loss_prior,
            "loss_recon": loss_recon,
            "loss_mixed": loss_mixed.mean(),
            "loss_masked": loss_masked.mean(),
        }
        model_stats = utils.loss2bpt(model_stats, self.data_shape)
        return model_stats

    def get_sampling_grid(self, i, timesteps):
        t = (timesteps - i) / timesteps
        s = t - 1 / timesteps
        if self.sampling_grid == "cosine":
            t = jnp.cos(math.pi / 2.0 * (1.0 - t))
            s = jnp.cos(math.pi / 2.0 * (1.0 - s))
        return s, t

    def ancestral_sample_step(self, rng, i, timesteps, zt, conditioning=None):
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        logits, _ = self.predict_x(zt, t, cond=cond)
        mean_preds = jax.nn.softmax(logits, axis=-1)

        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
        probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)

        to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_body)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)
        return zs

    def ancestral_sample_step_informed(self, rng, i, timesteps, zt, conditioning=None):

        B, D = zt.shape[:2]

        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        rng_pstep, rng_cstep = jr.split(rng_body, 2)

        # Predictor (ancestral)
        logits, _ = self.predict_x(zt, t, cond=cond)
        mean_preds = jax.nn.softmax(logits, axis=-1)

        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
        probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)

        to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_pstep)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)

        if self.k == 0:
            return zs

        # Corrector (gibbs)
        rng_cstep_1, rng_cstep_2 = jr.split(rng_cstep, 2)
        logits, _ = self.predict_x(zs, s, cond=cond)
        logits -= logsumexp(logits, axis=-1, keepdims=True)
        mean_preds = jax.nn.softmax(logits, axis=-1)

        jump_target = tfd.Categorical(probs=mean_preds).sample(seed=rng_cstep_1)
        # Figure out locations with the lowest score
        # Since the score is proportional to the denoising prob anyways, we're just gonna use the logits again
        b_idx, d_idx = jnp.indices((B, D))
        scores = logits[b_idx, d_idx, zs]
        # Add temperature annealing
        # This is minus since conventionally we add noise and take max
        scores -= self.gibbs_temp * jr.gumbel(rng_cstep_2, shape=(B, D))
        is_mask = zs == self.vocab_size
        scores = jnp.where(is_mask, jnp.inf, scores)
        
        # Trick: sort and then find the kth smallest
        thres = jnp.sort(scores, axis=-1)[:, self.k-1:self.k]
        zs = jnp.where((scores <= thres) & (zs != self.vocab_size), jump_target, zs)

        return zs

    def ancestral_sample_step_uninformed(self, rng, i, timesteps, zt, conditioning=None):

        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        rng_pstep, rng_cstep = jr.split(rng_body, 2)

        # Predictor (ancestral)
        logits, _ = self.predict_x(zt, t, cond=cond)
        mean_preds = jax.nn.softmax(logits, axis=-1)

        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
        probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)

        to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_pstep)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)

        # Corrector (uninformed)
        # Need to compute the backward rates from the logits
        # then sample with euler step...
        MASK = self.vocab_size

        logits, _ = self.predict_x(zs, s, cond=cond)
        logits -= logsumexp(logits, axis=-1, keepdims=True)
        # Shape: [B, D, S]
        mean_preds = jax.nn.softmax(logits, axis=-1)

        B, D, S = logits.shape
        b_idx, d_idx = jnp.indices((B, D))

        def _euler_update(key, x, rates):
            eps = 1e-8
            # Mask out the self transitions
            rates = rates.at[b_idx, d_idx, x].set(0.0)
            sum_rates = jnp.sum(rates, axis=-1)
            # transition_logit = jnp.log(-jnp.expm1(-rates)) # Prob = 1 - exp(-rate)
            transition_logit = jnp.log(-jnp.expm1(-sum_rates))[...,None] + jnp.log(rates) - jnp.log(sum_rates + eps)[...,None]
            transition_logit = transition_logit.at[b_idx, d_idx, x].set(-sum_rates)
            
            out = jr.categorical(key, transition_logit).astype(jnp.int32)
            return out

        dalpha_s = self.noise_schedule.dalpha(s)
        # Compute the rate matrix
        # Shape: [B, D, S+1]
        rates = jnp.where(zs[...,None] != MASK, 
            # Forward rate = - dalphat / alphat
            # Shape: [1, 1, S+1]
            jnp.concatenate([jnp.zeros((S,)), jnp.array((-dalpha_s / alpha_s,))])[None, None],
            # Backward rate = - dalphat / (1 - alphat) * denoising_probs
            # Shape: [B, D, S+1]
            jnp.concatenate([mean_preds * (-dalpha_s / (1-alpha_s)), jnp.zeros((B, D, 1))], axis=-1)
        )
        
        # The forward_backward_corrector shouldn't be used when s is 0
        zs = jax.lax.cond(s == 0, lambda x: x, 
            lambda x: _euler_update(rng_cstep, x, rates * self.uninformed_step_size * (t-s)), zs)

        return zs

    def maskgit_sample_step(self, rng, i, timesteps, zt, conditioning=None):

        B, D = zt.shape[:2]

        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        rng_1, rng_2 = jr.split(rng_body, 2)
        logits, _ = self.predict_x(zt, t, cond=cond)
        logits -= logsumexp(logits, axis=-1, keepdims=True)
        mean_preds = jax.nn.softmax(logits, axis=-1)

        jump_target = tfd.Categorical(probs=mean_preds).sample(seed=rng_1)
        # Figure out locations with the lowest score
        # Since the score is proportional to the denoising prob anyways, we're just gonna use the logits again
        b_idx, d_idx = jnp.indices((B, D))
        scores = logits[b_idx, d_idx, jump_target]
        # Add temperature annealing
        # This is minus since conventionally we add noise and take max
        scores += t * self.maskgit_temp * jr.gumbel(rng_2, shape=(B, D))
        # Don't touch tokens already generated
        scores = jnp.where(zt != self.vocab_size, -jnp.inf, scores)
        
        percentage_to_unmask = alpha_s - alpha_t
        # Shape: [B]
        mask_count = jnp.sum(zt == self.vocab_size, axis=-1)
        k = jnp.minimum(mask_count-1, jnp.floor(percentage_to_unmask * D).astype(int))
        k = jnp.maximum(1, k)

        # Trick: sort and then find the kth smallest
        thres = -jnp.sort(-scores, axis=-1)[jnp.arange(B), k-1][:, None]
        zt = jnp.where(scores >= thres, jump_target, zt)

        return zt

    def topp_sample_step(self, rng, i, timesteps, zt, conditioning=None, topp=0.98):
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        logits, _ = self.predict_x(zt, t, cond=cond)
        logits = binary_search.topp_mask(logits, topp, replace_val=jnp.array(-1e7))
        # mean_preds: [bs, ..., vocab]
        mean_preds = jax.nn.softmax(logits, axis=-1)

        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        probs_vocab = unmask_prob * mean_preds

        probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
        probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)

        to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_body)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)
        return zs

    def mean_sample_step(self, rng, i, timesteps, zt, conditioning=None):
        # Ancestral sampling done in two steps -- tends to be worse than one-step
        # implementation in ancestral_sample_step. See App. G of
        # https://arxiv.org/abs/2406.04329.
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        alpha_t = self.noise_schedule.alpha(t)
        alpha_s = self.noise_schedule.alpha(s)

        logits, _ = self.predict_x(zt, t, cond=cond)
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)

        rng_body, rng = jax.random.split(rng_body)
        z0 = tfd.Categorical(logits=logits).sample(seed=rng_body)

        rng_body, _ = jax.random.split(rng)
        unmask = jax.random.bernoulli(rng_body, unmask_prob, zt.shape)

        to_unmask = jnp.where(unmask, z0, zt)
        is_mask = zt == self.vocab_size
        zs = jnp.where(is_mask, to_unmask, zt)
        return zs

    def sample_step(self, rng, i, timesteps, zt, conditioning=None, topp=None, 
                    k=None, temp=None):
        if self.sampler == "ancestral":
            return self.ancestral_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        elif self.sampler == "gibbs":
            return self.ancestral_sample_step_informed(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        elif self.sampler == "maskgit":
            return self.maskgit_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        elif self.sampler == "uninformed":
            return self.ancestral_sample_step_uninformed(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        elif self.sampler == "topp":
            topp = self.topp if topp is None else topp
            return self.topp_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning, topp=topp
            )
        elif self.sampler == "mean":
            return self.mean_sample_step(
                rng, i, timesteps, zt, conditioning=conditioning
            )
        else:
            raise NotImplementedError()
