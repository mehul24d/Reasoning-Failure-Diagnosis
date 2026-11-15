# Distractor-clause review: agreement and changes

89 final clauses (A18), two reviewers who answered independently.

**Decision rule (union):** a clause is changed (rewritten or dropped) if EITHER reviewer marks it FIX, i.e. answers 'no' to is-it-a-no-op or to is-it-grammatical; agreement is reported, not used as a gate

| Question | Cohen's kappa | raw agreement | PABAK | reviewer 1 no/FIX | reviewer 2 no/FIX |
|---|---|---|---|---|---|
| is_noop | 0.232 | 0.787 | 0.573 | 10 | 19 |
| grammatical | -0.017 | 0.955 | 0.910 | 1 | 3 |
| verdict | 0.201 | 0.764 | 0.528 | 10 | 21 |

Marked FIX by either reviewer: [12, 14, 20, 22, 23, 27, 28, 35, 37, 39, 44, 47, 48, 51, 52, 56, 65, 67, 68, 69, 71, 81, 83, 92, 94, 97].

The changes below were agreed by the two reviewers **jointly**, in one discussion; they were not reviewed independently, so no agreement statistic applies to them.

Re-check after the review: all 86 clauses that stay in the study pass every automated check on instances 1-16 (66 with the independent-solver check).

* 12: drop (joint verdict 'Drop')
* 14: rewrite: "They pass {d} traffic lights on the way." -> "The town has {d} schools." (joint verdict 'Accept' (proposal: R2: traffic lights could add walking time; nothing on the route))
* 20: rewrite: "He started this job {d} weeks ago." -> "The farm has {d} tractors." (joint verdict 'Accept' (proposal: R1+R2: 19 weeks of the job is ambiguous with the fruit total; no time words))
* 22: rewrite: "She had invited {d} friends." -> "Her house has {d} windows." (joint verdict 'Accept' (proposal: R2: invited friends relate to the cutlery count; windows do not))
* 23: rewrite: "She has knitted {d} scarves for other relatives in the past." -> "Her house has {d} windows." (joint verdict 'Accept' (proposal: R2: scarves may share the wool; windows do not))
* 27: rewrite: "{d} of the emails that need a response come from the same client." -> "{name}'s office has {d} windows." (joint verdict 'Edit: Sanjay works for 9 hours everyday'; user decision 2026-11-15: the edit fails the no-forward-reference check ('work week' first appears in the final question) and d=17-19 gives implausible hours; replaced by the instance's name + office windows)
* 28: rewrite: "Each new patient waits about {d} minutes to be seen." -> "The building has {d} elevators." (joint verdict 'Accept' (proposal: R2: patient waiting time could read as delayed admissions; no time or patients))
* 35: rewrite: "He started raising money {d} days ago." -> "The venue has {d} exits." (joint verdict 'Accept' (proposal: R2: days of fundraising read as a rate; no time or money))
* 37: rewrite: "It was made {d} hours ago." -> "The recipe calls for {d} ingredients." (joint verdict 'Accept' (proposal: R1+R2: hours since it was made could shorten how long it lasts; no time))
* 39: drop (joint verdict 'Edit: The average age of maintainence workers is 45 years'; user decision 2026-11-15: the edit cannot fit d in 2-19 (an average age of 2-19 years) and 'average' first appears in the final question; dropped (train template))
* 44: drop (joint verdict 'Drop'; joint verdict Drop; leaves 39 test templates with a clause, gate lowered to 39 (S2 A21, D44))
* 47: rewrite: "The hunt started {d} minutes ago." -> "The yard has {d} trees." (joint verdict 'Accept' (proposal: R2: a hunt that started minutes ago is still going; no time))
* 48: rewrite: "She has been cooking for {d} years." -> "Her kitchen has {d} drawers." (joint verdict 'Accept' (proposal: R2: years of cooking could change her speed; no time or skill))
* 51: rewrite: "He has run the stand for {d} years." -> "The stand has {d} shelves." (joint verdict 'Accept' (proposal: R1: years running the stand; no time))
* 52: rewrite: "He has been saving for {d} weeks." -> "The shop has {d} aisles." (joint verdict 'Accept' (proposal: R1+R2: weeks of saving suggest more money; no time or money))
* 56: rewrite: "He bought the drinks {d} days ago." -> "{name} is {d} years old." (joint verdict 'Edit: Pavel is 23 years old'; name filled per instance (CLAUSE_SLOTS); 23 -> {d})
* 65: rewrite: "She has worked for herself for {d} years." -> "The {helper} lives {d} miles away." (joint verdict 'Edit: The bookkeeper lives 10 miles away'; role filled per instance (bookkeeper/accountant/tax consultant/financial advisor); 10 -> {d})
* 67: rewrite: "He is {d} kilometers from home." -> "The street has {d} streetlights." (joint verdict 'Accept' (proposal: R1: a distance from home could be read as work done pushing; no distance))
* 68: rewrite: "Her best friend played for {d} hours over the same period." -> "Her room has {d} posters on the wall." (joint verdict 'Accept' (proposal: R2: the best friend is a new person and 'she' became ambiguous; hours were the answer's unit))
* 69: rewrite: "He listens to {d} songs while he exercises." -> "The gym has {d} parking spaces." (joint verdict 'Accept' (proposal: R2: songs while exercising measure time; parking does not))
* 71: rewrite: "He has kept livestock for {d} years." -> "The barn has {d} windows." (joint verdict 'Accept' (proposal: R1: years keeping livestock; no time))
* 81: rewrite: "The twins arrived {d} minutes late." -> "The park has {d} gates." (joint verdict 'Accept' (proposal: R1: arriving late could mean fewer coins found, and the sentence read badly; no time))
* 83: rewrite: "They started baking {d} hours ago." -> "The {flavor} costs ${d}." (joint verdict 'Edit: The peanut butter costs $3.40'; flavour filled per instance (the 'some X' item); $3.40 -> ${d} (d is a whole number))
* 92: rewrite: "They fought for {d} hours." -> "The nearby castle had {d} {weapon}s." (joint verdict 'Edit: The nearby castle had 18 swords'; weapon filled per instance ('forgotten his X at home': sword/axe/hammer/mace); 18 -> {d})
* 94: rewrite: "{d} of the applicants live in another city." -> "The company has {d} offices." (joint verdict 'Accept' (proposal: R2: applicants from another city might not accept; no people))
* 97: rewrite: "The store has {d} other cash registers." -> "The store has {d} aisles." (joint verdict 'Accept' (proposal: R2: 'total weekly production' could include other registers; no registers or people))
