// Content for the Families chapter: every method on the map, hand-written.
//
// year   = year of the publication named in `origin` (the earliest one we are sure of; precursors
//          are listed in `famous`). Methods older than 1943 sit in a compressed strip on the timeline.
// signal = where the method's knowledge comes from (see SIGNALS). Where the fit is loose,
//          `signalNote` says why. Assigning a signal is an editorial judgement.
// importance 1-3 sets the node size (editorial: how central the method is to the field today).
// links  = ids of methods this one directly builds on (always older or of the same year).
// lab    = route(s) inside the atlas where the mechanism can be tried.

/** Families in display order (ring order in the tree, lane order on the timeline). */
export const FAMILIES = [
  { id: 'symbolic', name: 'Symbolic AI & search', short: 'Symbolic', color: '#8fa3c8', blurb: 'Knowledge written down as rules and logic; answers found by search.' },
  { id: 'classical', name: 'Classical ML', short: 'Classical ML', color: '#4d8dff', blurb: 'Statistical models that learn from tables of examples.' },
  { id: 'prob', name: 'Probabilistic models', short: 'Probabilistic', color: '#5ad7ff', blurb: 'Reasoning under uncertainty with explicit probability models.' },
  { id: 'neural', name: 'Neural architectures', short: 'Neural nets', color: '#4de3b0', blurb: 'The network designs that deep learning is built from.' },
  { id: 'repr', name: 'Representation learning', short: 'Representation', color: '#b6f36a', blurb: 'Learning useful vectors from raw, unlabeled data.' },
  { id: 'generative', name: 'Generative models', short: 'Generative', color: '#f3dd5e', blurb: 'Models that produce new text, images, audio and 3D scenes.' },
  { id: 'adapt', name: 'Adaptation & efficiency', short: 'Adaptation', color: '#ffb84d', blurb: 'Making a trained model fit a new job or a smaller machine.' },
  { id: 'systems', name: 'Compound AI systems', short: 'Compound systems', color: '#ff7a59', blurb: 'Models wired together with retrieval, tools and other models.' },
  { id: 'embodied', name: 'Perception & embodied AI', short: 'Perception', color: '#ff6fb1', blurb: 'Seeing, hearing and acting in the physical world.' },
  { id: 'rl', name: 'Reinforcement learning', short: 'Reinforcement', color: '#cf7dff', blurb: 'Learning to act from reward, by trial and error.' },
  { id: 'evo', name: 'Evolutionary & swarm', short: 'Evolutionary', color: '#9d8cff', blurb: 'Populations of candidates improved by variation and selection.' },
];

/** Learning signals: how a method gets its knowledge. */
export const SIGNALS = [
  { id: 'rules', name: 'Rules & search', blurb: 'No learning: people write the knowledge in as rules, models or prompts, and the machine computes the consequences.' },
  { id: 'supervised', name: 'Supervised', blurb: 'Learns from examples that come with the right answer attached.' },
  { id: 'unsupervised', name: 'Unsupervised', blurb: 'Finds structure in data that has no labels: groups, directions, densities.' },
  { id: 'self', name: 'Self-supervised', blurb: 'Hides part of the raw data and learns to predict it, so the data labels itself.' },
  { id: 'reinforcement', name: 'Reinforcement', blurb: 'Tries actions and learns from the reward that follows.' },
  { id: 'evolutionary', name: 'Evolutionary', blurb: 'Keeps a population of candidates and breeds the ones that score best.' },
];

/** Timeline annotations. Era boundaries are conventions, not exact dates. */
export const ERAS = [
  { id: 'winter1', from: 1974, to: 1980, name: 'First AI winter', note: 'Funding collapses after early promises go unmet (the Lighthill report appeared in 1973).' },
  { id: 'winter2', from: 1987, to: 1993, name: 'Second AI winter', note: 'The expert-system boom ends; the market for Lisp machines collapses in 1987.' },
  { id: 'deep', from: 2012, to: 2026, name: 'Deep-learning era', note: 'AlexNet wins the ImageNet challenge in 2012 and neural networks take over.' },
  { id: 'foundation', from: 2018, to: 2026, name: 'Foundation-model era', note: 'BERT and GPT (2018) establish pre-train once, adapt everywhere. The term dates from 2021.' },
];
export const MILESTONES = [{ year: 1956, name: 'Dartmouth workshop: the field is named' }];
export const YEAR_MIN = 1943;
export const YEAR_MAX = 2026;

/** Names of the atlas pages a method can link to. */
export const LAB_NAMES = {
  'labs/gpt': 'Inside a GPT',
  'labs/neural': 'Neural Network Lab',
  'labs/optim': 'Optimizer Arena',
  'labs/cnn': 'How a CNN Sees',
  'labs/diffusion': 'Diffusion Lab',
  'labs/world': 'Plan vs Learn',
  'labs/embed': 'Embedding Space',
  'perf/bakeoff': 'Model Bake-off',
  'perf/quant': 'Quantization experiment',
  'flow/rag': 'Data Flow: RAG',
  'flow/agent': 'Data Flow: agents and MCP',
};

const ARCH_NOTE = 'An architecture, not a training recipe: it can be trained with any signal. Shown here with the one it was introduced with.';

export const METHODS = [
  // ------------------------------------------------------------------ symbolic AI & search
  {
    id: 'minimax', name: 'Minimax & alpha-beta', family: 'symbolic', year: 1950, origin: 'Shannon’s paper on programming a computer to play chess', signal: 'rules', importance: 2,
    oneLine: 'Game-tree search that assumes the opponent always plays its best reply.',
    how: 'The program looks ahead through moves and counter-moves, scores the final positions with a hand-written evaluation function, and backs the scores up the tree: its own turns take the maximum, the opponent’s the minimum. Alpha-beta pruning skips branches that provably cannot change the result, so with good move ordering the same effort searches about twice as deep.',
    strengths: 'Exact within its search depth; the basis of decades of chess and checkers engines.',
    limits: 'Needs a good evaluation function and breaks down when there are too many possible moves, as in Go.',
    famous: ['Shannon, “Programming a Computer for Playing Chess” (1950)', 'Deep Blue beats Kasparov (1997)', 'Stockfish (2008)'],
    links: [],
  },
  {
    id: 'logic', name: 'Logic & SAT solving', family: 'symbolic', year: 1956, origin: 'Newell, Shaw and Simon’s Logic Theorist', signal: 'rules', importance: 2,
    oneLine: 'Derives conclusions from facts and rules by mechanical, checkable steps.',
    how: 'Knowledge is written as logical formulas. The program applies inference rules, or searches for an assignment of true and false that satisfies every formula. Modern SAT solvers learn a new constraint from every dead end, which lets them handle problems with millions of variables.',
    strengths: 'Answers come with proofs; the workhorse of chip and software verification.',
    limits: 'Everything must be stated formally, and the worst-case running time grows exponentially.',
    famous: ['Logic Theorist (1956)', 'DPLL procedure (1962)', 'Resolution, Robinson (1965)', 'Chaff SAT solver (2001)'],
    links: [],
  },
  {
    id: 'expert', name: 'Expert systems', family: 'symbolic', year: 1965, origin: 'The DENDRAL project at Stanford', signal: 'rules', importance: 3,
    oneLine: 'Hundreds of if-then rules captured from human specialists, applied by an inference engine.',
    how: 'Knowledge engineers interview experts and encode what they say as rules of the form “if these findings hold, then suspect that cause”. An inference engine chains the rules forward from the facts or backward from a hypothesis, and can show which rules led to its conclusion.',
    strengths: 'Transparent reasoning that can explain itself; works with no training data.',
    limits: 'Rules are costly to collect and maintain, and the system fails abruptly outside what it was told.',
    famous: ['DENDRAL (1965)', 'MYCIN (1970s)', 'XCON at Digital Equipment (1980)'],
    links: ['logic'],
  },
  {
    id: 'kg', name: 'Knowledge graphs', family: 'symbolic', year: 1966, origin: 'Quillian’s semantic networks', signal: 'rules', importance: 2,
    oneLine: 'Facts stored as a graph of entities linked by named relations.',
    how: 'Each fact is a triple such as (Marie Curie, born in, Warsaw). Queries follow the links. A schema, called an ontology, says which kinds of things and relations exist, so that new facts can be inferred from stored ones.',
    strengths: 'Precise, auditable facts that are easy to update; a reliable grounding source for language models.',
    limits: 'Building and curating the graph takes sustained human effort, and it only knows what was entered.',
    famous: ['Semantic networks, Quillian (1966)', 'Cyc (1984)', 'Google Knowledge Graph (2012)', 'Wikidata (2012)'],
    links: ['logic'],
  },
  {
    id: 'astar', name: 'A* search', family: 'symbolic', year: 1968, origin: 'Hart, Nilsson and Raphael, for Shakey the robot', signal: 'rules', importance: 3, lab: 'labs/world',
    oneLine: 'Finds the cheapest path by always expanding the most promising node first.',
    how: 'Every candidate node gets a score: the cost paid so far plus a heuristic estimate of the cost still to go. A* repeatedly expands the node with the lowest score. If the heuristic never overestimates, the first path that reaches the goal is guaranteed to be the cheapest.',
    strengths: 'Optimal and efficient when a good heuristic exists; standard in route planning and games.',
    limits: 'Needs a complete model of the world, and memory grows with the number of states explored.',
    famous: ['Hart, Nilsson & Raphael (1968)', 'Shakey the robot (1966–1972)'],
    links: [],
  },
  {
    id: 'planning', name: 'Automated planning', family: 'symbolic', year: 1971, origin: 'Fikes and Nilsson’s STRIPS planner', signal: 'rules', importance: 2,
    oneLine: 'Searches for a sequence of actions that turns the current state into a goal.',
    how: 'Each action is described by what must be true before it and what it changes. The planner searches through sequences of actions, guided by heuristics computed from a simplified version of the problem, until every goal condition holds.',
    strengths: 'General: one planner solves any problem written in its description language, and plans can be verified.',
    limits: 'Needs an accurate hand-written model of the actions; struggles with uncertainty and very large state spaces.',
    famous: ['STRIPS (1971)', 'Graphplan (1995)', 'PDDL planning language (1998)'],
    links: ['astar', 'logic'],
  },

  // ------------------------------------------------------------------ classical ML
  {
    id: 'linreg', name: 'Linear & logistic regression', family: 'classical', year: 1805, origin: 'Legendre’s method of least squares', signal: 'supervised', importance: 3, lab: 'perf/bakeoff',
    oneLine: 'A weighted sum of the inputs: a line for numbers, an S-curve for classes.',
    how: 'The model multiplies each input by a weight and adds the results. Training picks the weights that minimize the error on known examples: squared error for predicting numbers, log loss for probabilities. Logistic regression passes the sum through an S-shaped function to turn it into a probability.',
    strengths: 'Fast, stable and easy to interpret: each weight says how much one input matters.',
    limits: 'Only captures straight-line relationships unless you engineer the features yourself.',
    famous: ['Least squares, Legendre (1805) and Gauss (1809)', 'Logistic regression, Cox (1958)'],
    links: [],
  },
  {
    id: 'pca', name: 'PCA', family: 'classical', year: 1901, origin: 'Pearson, “On Lines and Planes of Closest Fit”', signal: 'unsupervised', importance: 2,
    oneLine: 'Finds the few directions along which the data varies the most.',
    how: 'Principal component analysis rotates the coordinate system so that the first axis points along the greatest spread of the data, the second along the greatest remaining spread, and so on. Keeping only the first few axes compresses the data while losing as little variance as possible.',
    strengths: 'Simple, exact and fast; a standard first step for visualization and noise removal.',
    limits: 'Linear only: it misses curved structure, and the components can be hard to interpret.',
    famous: ['Pearson (1901)', 'Hotelling (1933)', 'Eigenfaces for face recognition (1991)'],
    links: [],
  },
  {
    id: 'knn', name: 'k-nearest neighbors', family: 'classical', year: 1951, origin: 'Fix and Hodges’ technical report', signal: 'supervised', importance: 2, lab: 'perf/bakeoff',
    oneLine: 'Classifies a new point by a vote among its closest known examples.',
    how: 'There is no training step: the model simply stores every labeled example. To answer a query it measures the distance to the stored points, takes the k closest and returns their majority label or their average value.',
    strengths: 'No training and no assumptions about the shape of the data; easy to understand.',
    limits: 'Slow and memory-hungry at query time; distances lose meaning when many features are irrelevant.',
    famous: ['Fix & Hodges (1951)', 'Cover & Hart error bound (1967)'],
    links: [],
  },
  {
    id: 'kmeans', name: 'k-means clustering', family: 'classical', year: 1957, origin: 'Lloyd’s Bell Labs note (published in 1982)', signal: 'unsupervised', importance: 2,
    oneLine: 'Splits unlabeled data into k groups around k moving centers.',
    how: 'Start with k centers. Assign every point to its nearest center, move each center to the mean of the points assigned to it, and repeat until nothing changes. Each round lowers the total squared distance from points to their centers.',
    strengths: 'Simple and fast, even on very large data sets.',
    limits: 'You must choose k; the result depends on the starting centers and assumes round clusters of similar size.',
    famous: ['Lloyd’s algorithm (1957)', 'MacQueen coins “k-means” (1967)', 'k-means++ seeding (2007)'],
    links: [],
  },
  {
    id: 'nb', name: 'Naive Bayes', family: 'classical', year: 1961, origin: 'Maron’s probabilistic indexing of documents', signal: 'supervised', importance: 1, lab: 'perf/bakeoff',
    oneLine: 'Applies Bayes’ rule while pretending every feature is independent of the others.',
    how: 'From labeled examples it counts how often each feature appears in each class. For a new item it multiplies those per-feature probabilities with the class frequency and picks the most probable class. The independence assumption is usually false, yet the ranking of the classes is often still right.',
    strengths: 'Trains in one pass over the data, needs few examples and copes with very many features.',
    limits: 'Its probability estimates are poorly calibrated, and it cannot model interactions between features.',
    famous: ['Maron, “Automatic Indexing” (1961)', 'Bayesian spam filtering (1998)'],
    links: [],
  },
  {
    id: 'tree', name: 'Decision trees', family: 'classical', year: 1963, origin: 'Morgan and Sonquist’s AID procedure', signal: 'supervised', importance: 2, lab: 'perf/bakeoff',
    oneLine: 'A flowchart of yes/no questions learned from data.',
    how: 'The algorithm picks the single question, such as “age over 40?”, that best separates the training examples by outcome, splits the data, and repeats on each part. A prediction follows the questions from the root down to a leaf.',
    strengths: 'Readable, handles mixed data types and needs little preparation.',
    limits: 'A single tree overfits easily and changes a lot when the data changes a little.',
    famous: ['AID, Morgan & Sonquist (1963)', 'CART (1984)', 'ID3 (1986)', 'C4.5 (1993)'],
    links: [],
  },
  {
    id: 'svm', name: 'Support vector machines', family: 'classical', year: 1992, origin: 'Boser, Guyon and Vapnik’s kernel max-margin classifier', signal: 'supervised', importance: 2, lab: 'perf/bakeoff',
    oneLine: 'Draws the boundary that leaves the widest margin between two classes.',
    how: 'Among all boundaries that separate the classes, the SVM picks the one farthest from the nearest points of each class; only those nearest points, the support vectors, define it. The kernel trick computes similarities as if the data had been mapped into a much higher-dimensional space, so the boundary can curve.',
    strengths: 'Strong on small and medium data sets with many features; backed by solid theory.',
    limits: 'Training cost grows quickly with the number of examples; the kernel and its settings need tuning.',
    famous: ['Boser, Guyon & Vapnik (1992)', 'Soft-margin SVM, Cortes & Vapnik (1995)'],
    links: ['linreg'],
  },
  {
    id: 'cf', name: 'Collaborative filtering', family: 'classical', year: 1992, origin: 'The Tapestry mail system, which coined the term', signal: 'supervised', importance: 2,
    signalNote: 'The labels are ratings and clicks that users produce as a side effect.',
    oneLine: 'Recommends what people with similar tastes liked.',
    how: 'The data is a huge, mostly empty table of users and items. Neighborhood methods look for users with similar histories; matrix factorization gives every user and every item a short vector so that their dot product predicts the missing ratings.',
    strengths: 'Needs no description of the items, only behavior; scales to very large catalogs.',
    limits: 'Cold start: it has nothing to say about new users or items, and it reinforces what is already popular.',
    famous: ['Tapestry (1992)', 'GroupLens (1994)', 'Netflix Prize (2006–2009)'],
    links: ['knn', 'pca'],
  },
  {
    id: 'rf', name: 'Random forest', family: 'classical', year: 2001, origin: 'Breiman’s “Random Forests” paper', signal: 'supervised', importance: 2, lab: 'perf/bakeoff',
    oneLine: 'Hundreds of decision trees, each trained on a random slice of the data, vote.',
    how: 'Each tree is trained on a random sample of the examples and may only consider a random subset of the features at each split, so the trees make different mistakes. Averaging their votes cancels much of the individual error.',
    strengths: 'Accurate with little tuning, hard to overfit, and gives estimates of feature importance.',
    limits: 'Larger and slower than one tree, and no longer readable as a flowchart.',
    famous: ['Random decision forests, Ho (1995)', 'Bagging, Breiman (1996)', 'Random Forests, Breiman (2001)'],
    links: ['tree'],
  },
  {
    id: 'gbm', name: 'Gradient boosting', family: 'classical', year: 2001, origin: 'Friedman’s gradient boosting machine', signal: 'supervised', importance: 3, lab: 'perf/bakeoff',
    oneLine: 'Builds trees one after another, each correcting the errors of those before.',
    how: 'Start with a constant prediction. Each new small tree is fitted to the remaining errors, more precisely to the gradient of the loss, and added with a small weight. Hundreds of such steps turn many weak trees into one strong model.',
    strengths: 'Often the most accurate method on tabular data.',
    limits: 'Sensitive to its settings, can overfit noisy data, and trains sequentially.',
    famous: ['AdaBoost (1995)', 'Gradient boosting machine, Friedman (2001)', 'XGBoost (2016)', 'LightGBM (2017)'],
    links: ['tree'],
  },

  // @@MORE@@
];
