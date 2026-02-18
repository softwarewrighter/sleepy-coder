



You said:
Find and read recent Johns Hopkins ML papers on shared subspaces, and ELI5

ChatGPT said:
Here are two of the most recent and relevant machine-learning research papers from Johns Hopkins University on shared subspaces in deep learning — explained in a way that’s intuitive and grounded:

1. The Universal Weight Subspace Hypothesis (Dec 2025)
What the paper is about (in plain language):

Neural networks — like language models and vision models — have millions or billions of parameters. Just by intuition you’d expect that training different models on different tasks would make those parameters go in very different directions.

This paper shows something surprising: even when you train many different networks — different random starts, different data, different tasks — most of the variation ends up lying in the same small “geometric” piece of the parameter space. That piece is called a shared subspace, and it captures the main directions where meaningful learning happens. 

You can think of it like this: imagine a huge space with billions of dimensions. Instead of spreading out everywhere, all the learned networks collapse down near a common flat “sheet” — like different scribbles on a blackboard all stay near the same plane. Most of what matters about the network ends up in that plane. 

ELI5 Summary
Imagine every training run as a path through a huge jungle of possibilities.

The paper finds that almost all paths — whether you train a vision model, a language model, or something else — end up walking along the same narrow trail.

That narrow trail (the shared subspace) is much smaller than the full jungle (the full parameter space).

Because of this shared trail, a few directions capture most of what the network learns.

That means you could store and compute only those few directions instead of every parameter. 

Why it matters
Compression: Instead of storing billions of parameters, you might store only low-dimensional coordinates within this shared subspace. 

Fast adaptation to new tasks: You might be able to fine-tune with very few trainable parameters by staying within the shared subspace. 

Interpretability: It suggests that what neural networks really learn isn’t chaotic — it’s structured. 

Environmental impact: Less computation for training means lower energy use. 

2. Shared LoRA Subspaces for Almost Strict Continual Learning (Feb 2026)
What the paper is about (in plain language):

Adapting large models to new tasks in sequence is expensive and usually causes catastrophic forgetting — where the model forgets old tasks when learning new ones. Instead of training a full new model each time, people use LoRA adapters — small sets of trainable parameters — to tweak the model for a task.

This paper proposes Share, a new method that:

Learns a single shared subspace of adapter directions.

Incrementally updates that subspace as new tasks arrive.

Allows a model to learn many tasks without storing separate adapters for each one. 

It does this by treating the shared subspace like a “foundation” and only training a few coefficients for each new task within that shared subspace. 

ELI5 Summary
Imagine you have a bunch of Lego sets (tasks) and you use the same sort of bricks (shared subspace).

With traditional LoRA adapters, you make a new set of bricks for every new Lego model.

With Share, you keep one big box of core bricks and just tweak a few pieces to build new models.

That means you can build many models without needing a new huge bag of bricks each time. 

Why this is useful
Memory savings: Instead of 100+ adapters, you maintain one shared subspace with small updates. 

Better continual learning: New tasks don’t cause as much forgetting of prior ones. 

Efficiency: Less expensive to train and store task-specific information. 

Core Ideas Across Both Papers
At a conceptual level, these studies revolve around the insight that:

Neural network learning tends to concentrate in low-dimensional subspaces instead of filling all degrees of freedom.

These subspaces are shared across different tasks and models.

Leveraging this shared structure can enable efficient fine-tuning, compression, continued learning, and better understanding of model geometry. 

If you want, I can go a step deeper and summarize the mathematical approaches they use — including how they extract these subspaces (e.g., principal component analysis on weights) and how continual adaptation is implemented in the Share method.


You said:
What is an ideal number of LoRA adapters to implement shared subspaces and learning (frequently) without catastrophic forgetting

ChatGPT said:
You’re really asking two things hiding inside one:

How big should the shared subspace be?

How many LoRA adapters (or adapter directions) do I need so that I can keep learning frequently without blowing away old knowledge?

That’s a geometry question disguised as an engineering question.

Let’s unpack it carefully.

Step 1 — What are we actually controlling?
When you use LoRA, you are not adding “an adapter.” You are adding:

A low-rank update matrix ΔW

Factorized as A × B

With rank r

So what you’re really choosing is:

Rank per layer

Number of layers adapted

Number of tasks sharing the same subspace

In shared-subspace approaches (like the Johns Hopkins “Shared LoRA Subspaces” idea), you don’t want separate LoRAs per task forever. You want:

A shared basis (think: dictionary of directions)

Each task uses small coefficients within that basis

So the core control knob becomes:

The dimensionality of the shared subspace.

That’s your real “number of adapters.”

Step 2 — The geometric intuition (ELI5 but not fake-simple)
Imagine your 7B parameter model lives in a 7-billion-dimensional space.

But experiments suggest meaningful changes live in something like a 20–200 dimensional manifold.

So instead of giving each task its own new 16-rank LoRA, you:

Learn a shared 64–256 dimensional subspace

Express each task as coordinates inside it

That’s like saying:

Instead of giving every musician their own orchestra, give them the same orchestra and just change the sheet music.

Step 3 — So what’s an “ideal” number?
There is no universal constant. But we can reason from practice and theory.

For small-to-mid LLMs (1B–7B)
Typical stable numbers:

LoRA rank per layer: 8–64

Adapting 20–40% of layers

Effective subspace dimension: ~100–500 total

If you are building a shared subspace across many tasks:

Practical sweet spots:
Use case	Shared subspace size
5–10 related tasks	32–64
10–50 tasks	64–128
50–200 tasks	128–256
Aggressive continual learning	256–512
Above ~512, returns diminish unless tasks are wildly diverse.

Step 4 — Why too small is dangerous
If the shared subspace is too small:

New tasks collide in the same directions

Gradients overwrite prior task geometry

Catastrophic forgetting reappears

That’s just linear algebra:

If you only have 32 dimensions and 100 tasks, they will compete.

Step 5 — Why too large is also dangerous
If you make the shared subspace huge:

You recreate full fine-tuning

Memory grows

Training becomes unstable

You lose compression benefits

The magic is compression with separation.

Step 6 — A rule of thumb I’d actually use
For frequent learning cycles (what you want):

Start with:

Shared subspace = 128 dimensions

Then monitor:

Inter-task interference

Validation drift on old tasks

Cosine similarity between new task gradients and existing basis vectors

If:

Drift increases → expand subspace

Subspace remains underutilized → prune

Treat the subspace as elastic.

Step 7 — Better than picking a number: adaptive growth
The smartest design isn’t picking 128 or 256.

It’s:

Start small (64 or 128).

Track reconstruction error of new task gradients projected into subspace.

If projection error > threshold → add basis vector.

Periodically prune near-collinear directions.

That turns continual learning into dynamic basis management.

That’s far more powerful than a fixed “number of adapters.”

Step 8 — What catastrophic forgetting really is here
Forgetting happens when:

New task gradients lie outside the shared subspace
AND
You force them into the existing subspace

So your question becomes:

How often do my new tasks introduce fundamentally new directions?

If you are:

Doing coding tasks only → small subspace is fine

Mixing coding, math, biology, creative writing → larger subspace needed

Diversity drives dimension.

Step 9 — For your kind of experimental setup
Given your tendency to:

Fine-tune small 1–3B models

Iterate frequently

Measure progress

Visualize improvement over time

I would recommend:

Start at 128
Allow growth to 256
Cap at 512
Add pruning every N cycles

That will give you visible geometric behavior without wasting electricity.

Step 10 — The deeper truth
Here’s the weird part.

The ideal number isn’t about tasks.

It’s about:

How many independent conceptual directions your domain actually needs.

And that number is usually shockingly small.

Language models appear massively overparameterized.
But the useful directions seem low-dimensional.

That’s the big philosophical implication of the shared subspace hypothesis.

If you want, I can next:

Design a Rust+Python repo structure that simulates adaptive shared subspace growth

Show how to visualize basis expansion over 12 fine-tune cycles

Or propose an experiment using a 1B model on your 3090 to empirically estimate subspace size for your domain

There’s real experimental fun hiding in this geometry.





