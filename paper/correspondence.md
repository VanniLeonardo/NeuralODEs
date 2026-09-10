# Correspondence with the original authors

ReScience C expects a replication that reports a failure to have attempted contact with the
original authors. Claim 6 (Chen et al., Fig. 3c) does not reproduce, so that applies to us.
This file holds the drafts and the record of what was sent and what came back — including
"no reply", which is a perfectly acceptable outcome as long as the attempt is documented.

**Status: nothing sent yet.**

| Date sent | To | Subject | Reply |
|---|---|---|---|
| — | Chen et al. | Backward/forward NFE ratio in Fig. 3c | — |
| — | Dupont et al. | Table 1 training details | — |

## Addresses

Taken from the papers themselves, not from memory:

- **Chen et al. 2018** (arXiv 1806.07366): `rtqichen`, `rubanova`, `jessebett`, `duvenaud`
  — all `@cs.toronto.edu`, University of Toronto / Vector Institute.
- **Dupont et al. 2019** (arXiv 1904.01681): `dupont@stats.ox.ac.uk`,
  `doucet@stats.ox.ac.uk`, `y.w.teh@stats.ox.ac.uk`, University of Oxford.

⚠️ These are the 2018/2019 addresses. Several of these authors have moved institution since,
so expect at least one bounce. Address the email to the whole author list rather than one
person, and if it bounces, look up the current address rather than assuming the work is
unreachable — a bounce is not an attempt.

Email 1 is the one that matters for the submission requirement. Email 2 is optional but
genuinely useful: unstated training details are our leading explanation for the two accuracy
undershoots, so an answer could improve the replication rather than just satisfy a rule.

---

## Email 1 — Chen et al. (the claim that does not reproduce)

**To:** rtqichen@cs.toronto.edu, rubanova@cs.toronto.edu, jessebett@cs.toronto.edu, duvenaud@cs.toronto.edu
**Subject:** Question about the backward/forward NFE ratio in Fig. 3c of Neural ODEs

> Dear Dr Chen, Dr Rubanova, Dr Bettencourt and Prof Duvenaud,
>
> We are a group at Bocconi University preparing a replication of *Augmented Neural ODEs*
> (Dupont et al., 2019) for ReScience C. Because that paper builds on yours, our replication
> also tests four claims from *Neural Ordinary Differential Equations*. Three of them
> reproduce cleanly in our reimplementation: the solver error/cost trade-off of Fig. 3a–b,
> the growth of NFE over training in Fig. 3d, and the constant-memory property of the adjoint.
>
> One does not, and before we report that we would like to check whether we are measuring
> the same quantity you did. For Fig. 3c — backward NFE roughly half of forward NFE — we
> consistently measure a ratio far above one. On a trained convolutional ODE-Net on MNIST,
> at a tolerance where a forward/backward reconstruction check passes, we get a backward/forward
> ratio of about 123 with `dopri5` when the adjoint is solved at the same tolerance as the
> forward pass, and about 4 when the adjoint tolerance is set 100× looser (gradients agreeing
> with direct backpropagation to within 1% in both cases). We see the same pattern on 2-D toy
> fields.
>
> Three questions, if you have a moment:
>
> 1. In Fig. 3c, at what tolerance was the backward/adjoint solve run relative to the forward
>    pass — the same tolerance, or a looser or default one?
> 2. Was that figure produced with the implicit Adams solver used for the MNIST experiments,
>    and at the classification tolerance of 1e-3?
> 3. How is "backward NFE" counted — evaluations of `f` during the adjoint solve, or steps of
>    the augmented adjoint system?
>
> We ask because the adjoint tolerance alone accounts for most of the gap in our setup, and we
> would rather report a scope condition on the claim than a failure to reproduce it, if that is
> what it is.
>
> Our code and per-seed data are public at
> https://github.com/VanniLeonardo/NeuralODEs, and we are happy to share the draft. Any
> correction is welcome and will be reflected in the paper; we will note in the article if we
> did not manage to reach you.
>
> With thanks and best regards,
> Leonardo Vanni, on behalf of the authors
> Bocconi University

---

## Email 2 — Dupont et al. (optional; the two undershoots)

**To:** dupont@stats.ox.ac.uk, doucet@stats.ox.ac.uk, y.w.teh@stats.ox.ac.uk
**Subject:** Replication of Augmented Neural ODEs for ReScience C — two questions on Table 1

> Dear Dr Dupont, Prof Doucet and Prof Teh,
>
> We are a group at Bocconi University preparing a replication of *Augmented Neural ODEs* for
> ReScience C, reimplemented from the paper rather than from your released code. We are glad to
> report that the central claims reproduce: the 1-D crossing result holds in a strong form, with
> the Neural ODE sitting at the MSE floor of 1 that Proposition 1 predicts; NFE growth and the
> missing-slice generalisation gap both reproduce on two geometries.
>
> Two of the four matched-parameter accuracies in Table 1 come out below yours, and we would
> rather understand why than simply report a gap. At matched parameter counts and a
> reconstruction-checked solver tolerance, over five seeds, we get MNIST 94.2 ± 0.4 for the
> NODE (against 96.4 ± 0.5) and CIFAR-10 60.0 ± 0.9 for the ANODE (against 60.6 ± 0.4); the
> other two match closely and the ANODE-over-NODE ordering holds on both datasets.
>
> Our leading suspect is the training budget, which we could not find stated for the image
> experiments. If you can recall or still have to hand:
>
> 1. How many epochs were the MNIST and CIFAR-10 models trained for, and with which optimiser
>    and learning rate?
> 2. Were the Table 1 figures averaged over the same 20 repeats used for the toy experiments,
>    or fewer?
>
> Anything you can tell us will go into the paper, and we are of course happy to share the
> draft. Our code and per-seed results are at https://github.com/VanniLeonardo/NeuralODEs.
>
> With thanks and best regards,
> Leonardo Vanni, on behalf of the authors
> Bocconi University

---

## Notes on sending

- Send from an institutional address if you have one; a university address is far less likely
  to be filtered than a personal one for an unsolicited email to academics.
- Send **email 1 first**, and send it soon: it is the one tied to a submission requirement, and
  replies from busy academics take weeks. Nothing else in the submission depends on the answer.
- Do not chase more than once. One follow-up after two to three weeks is reasonable; after that,
  record the non-reply and move on — it does not weaken the submission.
- When a reply arrives (or does not), fill in the table at the top and write the outcome into
  the "Communication with the original authors" section of `content.tex`. Quote anything
  substantive, and if an author corrects us, say so plainly and fix the result.
- If email 1 bounces for everyone, a public issue on the `torchdiffeq` repository is a
  reasonable fallback, since the question is partly about that library's adjoint behaviour —
  but it is public, so keep it to the technical question.
