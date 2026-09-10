# Correspondence with the original authors

ReScience C expects a replication that reports a failure to have attempted contact with the
original authors. Claim 6, Chen et al. Figure 3c, does not reproduce, so this applies to us.
This file holds the drafts and the record of what was sent and what came back. No reply is an
acceptable outcome, provided the attempt is documented.

**Status: nothing sent yet.**

| Date sent | To | Subject | Reply |
|---|---|---|---|
| not yet | Chen et al. | Backward and forward NFE ratio in Fig. 3c | |
| not yet | Dupont et al. | Table 1 training details | |

## Addresses

Taken from the papers themselves.

- **Chen et al. 2018** (arXiv 1806.07366): `rtqichen`, `rubanova`, `jessebett` and
  `duvenaud`, all at `@cs.toronto.edu`, University of Toronto and the Vector Institute.
- **Dupont et al. 2019** (arXiv 1904.01681): `dupont@stats.ox.ac.uk`,
  `doucet@stats.ox.ac.uk`, `y.w.teh@stats.ox.ac.uk`, University of Oxford.

These are the 2018 and 2019 addresses. Several of these authors have moved institution since,
so expect at least one bounce. Write to the whole author list rather than to one person. If an
address bounces, look up the current one. A bounce is not an attempt.

Email 1 is the one the submission requirement rests on. Email 2 is optional and useful on its
own terms. Unstated training details are our leading explanation for the two accuracy
undershoots, so an answer would improve the replication.

---

## Email 1: Chen et al., the claim that does not reproduce

**To:** rtqichen@cs.toronto.edu, rubanova@cs.toronto.edu, jessebett@cs.toronto.edu, duvenaud@cs.toronto.edu

**Subject:** Question about the backward and forward NFE ratio in Fig. 3c of Neural ODEs

> Dear Dr Chen, Dr Rubanova, Dr Bettencourt and Prof Duvenaud,
>
> We are a group at Bocconi University preparing a replication of *Augmented Neural ODEs*
> (Dupont et al., 2019) for ReScience C. Because that paper builds on yours, our replication
> also tests four claims from *Neural Ordinary Differential Equations*. Three of them
> reproduce cleanly in our reimplementation: the error and cost trade-off of Fig. 3a and 3b,
> the growth of NFE over training in Fig. 3d, and the constant memory of the adjoint.
>
> One does not, and before we report that we would like to check whether we are measuring the
> same quantity you did. Fig. 3c reports a backward pass costing roughly half the forward pass
> in function evaluations. We consistently measure a ratio far above one. On a trained
> convolutional ODE-Net on MNIST, at a tolerance where a forward and backward reconstruction
> check passes, the ratio is about 123 with `dopri5` when the adjoint is solved at the same
> tolerance as the forward pass, and about 4 when the adjoint tolerance is set 100 times
> looser. In both cases the gradients agree with direct backpropagation to within 1%. We see
> the same pattern on two-dimensional toy fields.
>
> Three questions, if you have a moment:
>
> 1. In Fig. 3c, at what tolerance was the adjoint solve run relative to the forward pass? The
>    same tolerance, or a looser or default one?
> 2. Was that figure produced with the implicit Adams solver used for the MNIST experiments,
>    and at the classification tolerance of 1e-3?
> 3. How is the backward NFE counted? Evaluations of `f` during the adjoint solve, or steps of
>    the augmented adjoint system?
>
> We ask because the adjoint tolerance alone accounts for most of the gap in our setup. If that
> is what explains it, we would rather report a scope condition on the claim than a failure to
> reproduce it.
>
> Our code and per-seed data are public at https://github.com/VanniLeonardo/NeuralODEs, and we
> are happy to share the draft. Any correction is welcome and will be reflected in the paper.
> We will note in the article if we did not manage to reach you.
>
> With thanks and best regards,
> Leonardo Vanni, on behalf of the authors
> Bocconi University

---

## Email 2: Dupont et al., the two undershoots (optional)

**To:** dupont@stats.ox.ac.uk, doucet@stats.ox.ac.uk, y.w.teh@stats.ox.ac.uk

**Subject:** Replication of Augmented Neural ODEs for ReScience C, two questions on Table 1

> Dear Dr Dupont, Prof Doucet and Prof Teh,
>
> We are a group at Bocconi University preparing a replication of *Augmented Neural ODEs* for
> ReScience C, reimplemented from the paper rather than from your released code. We are glad to
> report that the central claims reproduce. The one-dimensional crossing result holds in a
> strong form, with the Neural ODE sitting at the MSE floor of 1 that Proposition 1 predicts.
> The NFE growth and the missing-slice generalisation gap both reproduce on two geometries.
>
> Two of the four matched-parameter accuracies in Table 1 come out below yours, and we would
> rather understand why than simply report a gap. At matched parameter counts and a
> reconstruction-checked solver tolerance, over five seeds, we get 94.2 ± 0.4 for the Neural
> ODE on MNIST, against your 96.4 ± 0.5, and 60.0 ± 0.9 for the augmented model on CIFAR-10,
> against your 60.6 ± 0.4. The other two match closely, and the ordering holds on both
> datasets.
>
> Our leading suspect is the training budget, which we could not find stated for the image
> experiments. If you can recall or still have to hand:
>
> 1. How many epochs were the MNIST and CIFAR-10 models trained for, and with which optimiser
>    and learning rate?
> 2. Were the Table 1 figures averaged over the same 20 repeats used for the toy experiments,
>    or fewer?
>
> Anything you can tell us will go into the paper, and we are happy to share the draft. Our
> code and per-seed results are at https://github.com/VanniLeonardo/NeuralODEs.
>
> With thanks and best regards,
> Leonardo Vanni, on behalf of the authors
> Bocconi University

---

## Notes on sending

- Send from an institutional address if you have one. A university address is much less likely
  to be filtered than a personal one for unsolicited academic email.
- Send email 1 first, and soon. It is the one tied to a submission requirement, and replies
  from busy academics take weeks. Nothing else in the submission depends on the answer.
- Do not chase more than once. One follow-up after two or three weeks is reasonable. After
  that, record the non-reply and move on. It does not weaken the submission.
- When a reply arrives, or does not, fill in the table above and write the outcome into the
  "Communication with the original authors" section of `content.tex`. Quote anything
  substantive. If an author corrects us, say so plainly and fix the result.
- If email 1 bounces for every address, a public issue on the `torchdiffeq` repository is a
  reasonable fallback, since the question is partly about that library's adjoint behaviour. It
  is public, so keep it to the technical question.
