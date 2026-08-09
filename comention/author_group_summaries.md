# Author co-mention groups — what each one is

Interpretation of the clusters in `author_groups.txt`, produced by
`comention.py` on the self-mention-free extraction of 2026-07 (a blurb
name-dropping its own book's author no longer counts — that was 40% of raw
mentions). Two authors are linked when the same blurb name-drops both.

**Method (changed 2026-07):** single-pass **Leiden** community detection
(Traag et al. 2019, RB modularity) at resolution γ = 8, chosen so the largest
community stays readable (~180 authors) while 94% of graph authors still land
in a reportable group. This replaced Louvain + recursive re-splitting of
oversized groups: one principled knob instead of three ad-hoc ones, plus
Leiden's guarantee that communities are internally well-connected. The
partition is **flat** — there are no sub-groups anymore; what used to be
"leaves" of two giant genre blobs are now top-level groups in their own
right. **201 groups total.**

**This file is load-bearing:** `comention.py` parses the bold-titled bullets
and the table rows below back out of it and uses them as the group titles in
`comention_groups.png` and `author_groups.csv`.
Keep the `**G12 (101) Title**` / `| G64 | 37 | Title |` shapes when editing;
an untitled group falls back to its lead authors.

Reading notes:
- These clusters reflect **who blurb-writers compare you to**, not literary
  influence. The connectors are anthology tables of contents, award names,
  and "for fans of X and Y" formulas — so clusters land on marketing scenes,
  genres, and eras.
- At this resolution the old genre continents (canon, SF/F/H, crime…) appear
  as *several* adjacent groups each — e.g. the classic canon is sliced into
  era/nationality communities (Romantics G44, Bloomsbury G70, Austen's Britain
  G86, Irish modernists G90, transcendentalists G94, French G95, Russians
  G100, German-language G57…). The overview figure shows how they touch.
- Any modularity-style partition here is reproducible for fixed input but
  **not stable under small perturbations** — collapsing one duplicate name
  moved boundary authors between adjacent communities. Trust the community, not the
  exact border.

---

## The big communities (G0–G30)

- **G0 (177) Contemporary American poetry** — Billy Collins, Ashbery, Adrienne Rich, Rita Dove, Mary Oliver, Sharon Olds. The whole living-poets circuit in one group.
- **G1 (158) Thriller bestsellers** — Lee Child, Connelly, Rollins, Patterson, Clancy, Dan Brown. Techno, legal, forensic and political thriller in one machine.
- **G2 (151) Comics: canon & superhero** — Gaiman, Kirby, Stan Lee, Alan Moore, Grant Morrison, Eisner.
- **G3 (139) Christianity** — Aquinas, Merton, Jonathan Edwards, Francis of Assisi, Billy Graham. Catholic mystics through evangelical celebrity.
- **G4 (136) Rock & pop memoir** — Springsteen, Jagger, Patti Smith, McCartney, Lou Reed, Cash.
- **G5 (122) Creative-writing (MFA) circuit** — Aimee Bender, Jim Shepard, Koestenbaum, Charles Baxter, Russo. The writers who teach and are taught on the Master of Fine Arts circuit: workshop short fiction + the essay world.
- **G6 (120) Self-help & spirituality** — Thich Nhat Hanh, Chopra, Ram Dass, Napoleon Hill, Tony Robbins. Buddhism and success literature share blurbs.
- **G7 (117) Comedy & punditry** — Ephron, Steve Martin, Mel Brooks, Lenny Bruce + Limbaugh, O'Reilly, Franken. Humor essay and political shout-books.
- **G8 (109) Continental philosophy & theory** — Camus, Beauvoir, Benjamin, Sartre, Derrida, Arendt.
- **G9 (108) Contemporary romance** — Nora Roberts, Mallery, Shalvis, Macomber, Sparks.
- **G10 (105) Literary & small-press horror** — Ramsey Campbell, Matheson, Lansdale, Straub, Ligotti.
- **G11 (104) American crime: private-eye tradition** — Elmore Leonard, Block, Grafton, Lehane, McBain, Paretsky.
- **G12 (101) Golden Age detection** — Conan Doyle, Christie, Sayers, Rendell, Highsmith, Chesterton.
- **G13 (96) Food writing** — Trillin, Julia Child, Waters, Bourdain, Reichl, Pollan.
- **G14 (95) The great-books canon** — Adam Smith, Nietzsche, Kant, Mill, Hume, Wittgenstein. Philosophy + political economy invoked as pedigree.
- **G15 (94) Golden-age Hollywood** — Chaplin, Brando, Bette Davis, Dietrich, Olivier.
- **G16 (92) New-century SF/fantasy** — Kress, Sterling, Kelly Link, Swanwick, Doctorow, VanderMeer. The Asimov's/anthology generation and its genre-benders.
- **G17 (90) Literary SF elders** — Le Guin, Silverberg, Gene Wolfe, Benford, Card, Joanna Russ.
- **G18 (90) Small-press extreme horror** — Maberry, Keene, Braunbeck, Ketchum. ("Bram Stoker" sits here as the award hub.)
- **G19 (87) Popular science & public intellectuals** — Hawking, Hitchens, Gould, Chomsky, Sagan, Pinker.
- **G20 (85) Golden Age SF** — Leiber, Poul Anderson, Asimov, Sturgeon, Clarke, Heinlein.
- **G21 (84) Mythic & anthology fantasy** — Tanith Lee, de Lint, McCaffrey, Yolen, Marion Zimmer Bradley, Lackey. The Datlow–Windling world.
- **G22 (83) Civil-rights & inspirational memoir** — Helen Keller, Steinem, Paine, Rosa Parks + the founders. School-anthology pedigree names.
- **G23 (83) Nature & environmental writing** — Dillard, Rachel Carson, Muir, Wendell Berry, Abbey, McPhee.
- **G24 (81) Alternative/literary comics** — Spiegelman, Ware, Clowes, Crumb, Lynda Barry.
- **G25 (80) Humor & memoir-essay** — Sedaris, Keillor, Dave Barry, Mary Karr, Lamott.
- **G26 (77) Art & photography** — Kahlo, Cartier-Bresson, Avedon, Rivera, O'Keeffe, Leibovitz.
- **G27 (74) The Black canon** — Langston Hughes, Morrison, Alice Walker, Gwendolyn Brooks, Hurston, Douglass.
- **G28 (74) Baseball (+ wrestling)** — Mays, Mantle, Ted Williams, Aaron, Yogi Berra.
- **G29 (72) Postmodern & maximalist fiction** — DeLillo, DFW, Franzen, Chabon, Pynchon, Zadie Smith.
- **G30 (70) Brooklyn/pop-culture essayists** — Lethem, Eggers, Orlean, Greil Marcus, Klosterman.

## The mid-size communities (G31–G60)

- **G31 (70)** Big-house US literary — Oates, Russell Banks, Prose, Styron, Smiley.
- **G32 (69)** Film directors — Scorsese, Lynch, Kubrick, Tarantino. (The "John Williams" composer/novelist collision sits here.)
- **G33 (65)** International prize circuit — Rushdie, Angela Carter, Lessing, Murakami, McEwan, Mantel.
- **G34 (64)** Paranormal romance & urban fantasy — Butcher, Charlaine Harris, Nalini Singh, J.R. Ward + Regency (Quinn, Balogh).
- **G35 (62)** Women's commercial fiction — Gillian Flynn, Kinsella, Moyes, Giffin, Weiner. Chick lit + domestic suspense.
- **G36 (61)** The Inklings & Christian fantasy — Lewis, Tolkien, Pratchett, Douglas Adams, Rowling, Pullman.
- **G37 (60)** Canadian literature — Atwood, Munro, Gallant, Robertson Davies, Ondaatje.
- **G38 (59)** The Beats — Ginsberg, Kerouac, Burroughs, Snyder, Bukowski, Kesey.
- **G39 (59)** Epic fantasy & YA blockbusters — George R.R. Martin (now a single node), Gabaldon, Hobb, Robert Jordan, Suzanne Collins, Riordan.
- **G40 (58)** Latin American letters — Borges, Neruda, Paz, Vargas Llosa, Fuentes, Rulfo.
- **G41 (58)** Minimalist & experimental short story — Carver, Lorrie Moore, Lydia Davis, Saunders, Barthelme.
- **G42 (55)** The stage — Beckett, Tennessee Williams, Arthur Miller, O'Neill, Albee, Pinter.
- **G43 (54)** Travel writing — Jan Morris, Iyer, Theroux, Bryson (+ stray realists).
- **G44 (50)** The Romantics & the English poetic canon — Shakespeare, Keats, Wordsworth, Blake, Donne, Milton.
- **G45 (49)** Basketball & action-star memoir — Jordan, Kareem, Schwarzenegger, Shaq.
- **G46 (47)** UK procedural & tartan noir — Rankin, Peter Robinson, Billingham, McDermid (+ a Chaucer artifact via "Canterbury" blurbs).
- **G47 (47)** Multicultural American fiction — Alexie, Cisneros, Amy Tan, Erdrich, Silko, Kincaid.
- **G48 (47)** Bestseller horror — Stephen King, Koontz, Barker, Anne Rice, Joe Hill.
- **G49 (46)** Experimental & feminist poetics — Mullen, Susan Howe, Kapil, Bernstein, Brossard.
- **G50 (45)** The classic weird tale — Bierce, Washington Irving, Maupassant, Blackwood, Machen, M.R. James.
- **G51 (43)** Children's & crossover UK — Dahl, Snicket, Blume, Cleary (+ Nabokov by blurb adjacency).
- **G52 (42)** Southern & grit lit — Cormac McCarthy, Larry Brown, Harry Crews, Ron Rash.
- **G53 (42)** South Asian literature — Lahiri, Desai, Roy, Seth, Narayan, Tagore-adjacent moderns.
- **G54 (42)** The New Yorker institution — Updike, Gladwell, Kael, Remnick, Angell.
- **G55 (42)** Mid-century Britain — Orwell, Greene, Waugh, Huxley, Burgess.
- **G56 (42)** Doctor Who/Torchwood + kids' fiction — Barrowman, Russell T. Davies + Walter Dean Myers, Jeff Kinney. (Two thin communities fused; treat separately.)
- **G57 (40)** German-language modernists — Kafka, Mann, Brecht, Hesse, Zweig, Musil.
- **G58 (40)** New Journalism & literary celebrity — Capote, Mailer, Didion, Tom Wolfe, Vidal, Hunter S. Thompson.
- **G59 (40)** Historical fiction: Rome to Tudor — Cornwell, Iggulden, Scarrow, Philippa Gregory, Chevalier.
- **G60 (40)** Statesmen & Cold War — Kissinger, Powell, Eisenhower, Nixon.

## Era-sliced canon (the old G0/G7 giants, now standalone communities)

G61–G130 are dominated by fine slices of the classic shelf; highlights:

| Group | Size | Community |
|---|---|---|
| G64 | 37 | Paris avant-garde salon (Stein, Duchamp, Man Ray, Cocteau) |
| G65 | 37 | Hardboiled classics (Chandler, Hammett, Cain, Jim Thompson) |
| G66 | 36 | Children's classics & nonsense (Carroll, Lear, Andersen, Baum, Milne) |
| G68 | 34 | Jewish-American fiction & NY intellectuals (Roth, Bellow, Sontag, Paley) |
| G69 | 34 | Irish & UK short story (Trevor, Frank O'Connor, Edna O'Brien) |
| G70 | 33 | Bloomsbury & interwar Britain (Woolf, Mansfield, Forster, Rhys) |
| G71 | 33 | Business titans & investing (Ford, Buffett, Carnegie + Turing oddly) |
| G72 | 33 | American modernist poetry (Pound, Stevens, W.C. Williams, Moore) |
| G73 | 31 | Black postwar prose (Baldwin, Wright, Ellison) + midcentury journalists |
| G76 | 29 | Lost Generation & its critics (Fitzgerald, Edmund Wilson, Mencken) |
| G78 | 28 | Irish poetry (Yeats, Heaney, Swift, Walcott) |
| G79 | 28 | Confessionals (Plath, Bishop, Lowell, Sexton, Ted Hughes) |
| G80 | 27 | Italian letters (Calvino, Eco, Primo Levi, Dante) |
| G81 | 26 | Monty Python & UK comedy (Cleese, Fry, Clarkson) |
| G83 | 25 | Empire adventure (Kipling, Jack London, Haggard) |
| G85 | 25 | Jazz & soul (Ellington, Armstrong, Miles, Holiday) |
| G86 | 24 | Austen's Britain (Austen, George Eliot, Trollope, Heyer) |
| G88 | 24 | American schoolroom & children's verse (Frost, Sandburg, Silverstein) |
| G89 | 22 | Weird Tales pulps (Lovecraft, R.E. Howard, Clark Ashton Smith, Derleth) |
| G90 | 22 | Irish modernists (Joyce, Flann O'Brien, Ford Madox Ford) |
| G91 | 21 | European lyric in translation (Rilke, Akhmatova, Celan, Brodsky) |
| G92 | 21 | Dance & modernist music (Balanchine, Graham, Stravinsky) |
| G93 | 21 | Early detection (Poe as hub, Anna Katharine Green, Futrelle) |
| G94 | 21 | Transcendentalists & 19th-c New England (Emerson, Thoreau, Alcott, Stowe) |
| G95 | 20 | French 19th century (Flaubert, Proust, Hugo, Baudelaire) |
| G100 | 19 | Russians (Tolstoy, Chekhov, Dostoyevsky) |
| G101 | 19 | Henry James's circle (James, Wharton, Henry Adams) |
| G103 | 19 | Algonquin wits (Dorothy Parker, Ogden Nash, Groucho, Lebowitz) |
| G104 | 18 | War poets (Graves, Owen, Sassoon + Hardy) |
| G106 | 18 | Southern fiction classic (Faulkner, O'Connor, Welty, McCullers) |
| G113 | 15 | American naturalists (Cather, Crane, Chopin, Dreiser) |
| G114 | 14 | Transgressive contemporary (Palahniuk, Ellis, Welsh, Houellebecq) |
| G117 | 13 | Westerns (L'Amour, Zane Grey) |
| G118 | 12 | Transgressive lineage (Henry Miller, Genet, de Sade, Acker) |
| G123 | 11 | Scientific romance (Wells, Verne, Burroughs — "John Carter" collision here) |
| G125 | 10 | War correspondents (Hemingway, Gellhorn, Ernie Pyle, Shirer) |

## The tail (G126–G200, 3–10 authors each)

~75 micro-communities, each one hobby, imprint, or national literature: UK football
(G126, G194), Lonely Planet (G127), NFL (G128), cozy-mystery lines (G131,
G167, G184, G200), picture books (G133, G190), cyberpunk splinter (G134),
Doctor Who audio (G135, G157), evangelical commentary (G136, G143, G161),
ancient-aliens esoterica (G138), Norwegian literature (G139), knitting
(G140), Dutch literature (G191), contemporary Chinese fiction (G197), Mexican
fiction (G147), Lewis & Clark (G149), Civil War historians (G150), vegan diet
(G152), poker (G154), chess (G164), running (G166), Alcoholics Anonymous
founders (G178), narrative history (G188), UML/software engineering (G189),
Formula 1 (G162), cricket (G186), Franco-Belgian comics — bande dessinée —
(G187), Amish romance (G177), Harlequin category
lines (G121, G148, G198, G199). Real but tiny: pairs recur in each other's
blurbs and nowhere else.