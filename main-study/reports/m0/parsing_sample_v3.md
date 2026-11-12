# Random parsed steps (S1c traces) for manual checking

## gsm8k-train-647 terse_p0 trace_correct=True
* step text: `* **Cupcakes left:** 40 cupcakes - 38 cupcakes = 2 cupcakes`
* parsed: op=- a=40 b=38 result=2 a_ref=('slot', 0) b_ref=('const', Fraction(38, 1)) compound=False ambiguous=False
* aligned gold step: 5 (tier 3) (-, ('res', 4), ('const', Fraction(1, 1)), 2)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 40 - 18 = 22; 1: 22 - 16 = 6; 2: 6 - 1 = 5; 3: 5 - 1 = 4; 4: 4 - 1 = 3; 5: 3 - 1 = 2
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-980 terse_p0 trace_correct=True
* step text: `* Total corn cobs: 16 * 4 = 64 corn cobs`
* parsed: op=* a=16 b=4 result=64 a_ref=('slot', 2) b_ref=('slot', 3) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('slot', 2), ('slot', 3), 64)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 13 * 4 = 52; 1: 16 * 4 = 64; 2: 52 + 64 = 116
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-944 marked_p0 trace_correct=True
* step text: `Step 5: Daisies left: 14 - 7 = 7`
* parsed: op=- a=14 b=7 result=7 a_ref=('res', 0) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=True  rederived=False
* gold program: 0: 5 + 9 = 14; 1: 14 / 2 = 7
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4125 marked_p0 trace_correct=True
* step text: `Step 7:  Solve for x: x = 9 days * (4/3) = 12 days`
* parsed: op=None a=None b=None result=12 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 2 (tier 3) (*, ('res', 1), ('res', 0), 12)
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 7 + 2 = 9; 1: 4 / 3 = 4/3; 2: 4/3 * 9 = 12; 3: 12 / 2 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1792 terse_p0 trace_correct=True
* step text: `* **Calculate total flowers per table:** Each table needs 8 roses + 12 daisies + 3 snapdragons + 6 lilies = 29 flowers.`
* parsed: op=None a=None b=None result=29 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 3 (tier 3) (+, ('res', 2), ('res', 0), 29)
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 2 * 3 = 6; 1: 8 + 12 = 20; 2: 20 + 3 = 23; 3: 23 + 6 = 29; 4: 29 * 10 = 290
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-7454 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total time spent grooming in 30 days.
   Total time = 40 minutes/day * 30 days = 1200 minutes`
* parsed: op=* a=40 b=30 result=1200 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('slot', 2), ('res', 0), 1200)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2 * 20 = 40; 1: 30 * 40 = 1200; 2: 1200 / 60 = 20
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3572 marked_p0 trace_correct=False
* step text: `Step 4: Add the number of blouses and dresses. 8 blouses + 6 dresses = 14 pieces of clothes`
* parsed: op=+ a=8 b=6 result=14 a_ref=('res', 1) b_ref=('rres', 2) compound=False ambiguous=False
* aligned gold step: 4 (tier 4) (+, ('res', 1), ('res', 3), 17)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 2 * 60 = 120; 1: 120 / 15 = 8; 2: 3 * 60 = 180; 3: 180 / 20 = 9; 4: 8 + 9 = 17
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2751 terse_p0 trace_correct=True
* step text: `* **Calculate the total number of cards:**  The total number of cards is 20 (Carlos) + 14 (Matias) + 14 (Jorge) = 48 cards.`
* parsed: op=None a=None b=None result=48 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 2 (tier 3) (+, ('res', 1), ('slot', 1), 48)
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 20 - 6 = 14; 1: 14 + 14 = 28; 2: 28 + 20 = 48
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6783 marked_p0 trace_correct=False
* step text: `Step 1: Calculate the total cost of one sandwich and one pack of juice: $0.30 + $0.20 = $0.50`
* parsed: op=+ a=3/10 b=1/5 result=1/2 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (+, ('slot', 0), ('slot', 1), 1/2)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 3/10 + 1/5 = 1/2; 1: 5/2 / 1/2 = 5; 2: 5 - 1 = 4
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2356 terse_p0 trace_correct=False
* step text: `* Apples were two times cheaper than oranges, so they cost $12 / 2 = $6 per kg.`
* parsed: op=/ a=12 b=2 result=6 a_ref=('slot', 2) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (/, ('slot', 2), ('const', Fraction(2, 1)), 6)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 12 / 2 = 6; 1: 6 / 3 = 2
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6794 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the cost of Francis's muffins.
Cost of 2 muffins = 2 * $2 = $4`
* parsed: op=* a=2 b=2 result=4 a_ref=('slot', 0) b_ref=('slot', 0) compound=False ambiguous=True
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('slot', 0), 4)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2 * 2 = 4; 1: 2 * 3 = 6; 2: 2 * 2 = 4; 3: 1 * 3 = 3; 4: 4 + 6 = 10; 5: 10 + 4 = 14; 6: 14 + 3 = 17
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-374 terse_p0 trace_correct=True
* step text: `* **Yesterday's buyers:** The day before had 50 buyers, and yesterday had half that number, so yesterday had 50 / 2 = 25 buyers.`
* parsed: op=/ a=50 b=2 result=25 a_ref=('slot', 1) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('const', Fraction(1, 2)), ('slot', 1), 25)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 1/2 * 50 = 25; 1: 40 + 25 = 65; 2: 65 + 25 = 90; 3: 90 + 50 = 140
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6420 marked_p0 trace_correct=False
* step text: `Step 9: Calculate the total distance Rich walked: 20 feet + 200 feet + 440 feet + 330 feet = 990 feet.`
* parsed: op=None a=None b=None result=990 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 4 (tier 3) (+, ('res', 2), ('res', 3), 990)
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 20 + 200 = 220; 1: 220 * 2 = 440; 2: 220 + 440 = 660; 3: 660 / 2 = 330; 4: 660 + 330 = 990; 5: 990 * 2 = 1980
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1666 marked_p0 trace_correct=False
* step text: `Step 2: Calculate the cost for 4 weeks: $0.10 per day * 7 days/week * 4 weeks = $2.80`
* parsed: op=None a=None b=None result=14/5 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 1/2 * 5 = 5/2; 1: 5/2 * 4 = 10
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4870 terse_p0 trace_correct=True
* step text: `* Revenue from children's tickets = 28 * $13 = $364`
* parsed: op=* a=28 b=13 result=364 a_ref=('slot', 5) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (*, ('slot', 5), ('res', 1), 364)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 183 * 26 = 4758; 1: 26 / 2 = 13; 2: 28 * 13 = 364; 3: 4758 + 364 = 5122
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-7140 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the sunflower's height.
   Sunflower height = Sister's height + 21 inches
   Sunflower height = 51 inches + 21 inches = 72 inches`
* parsed: op=+ a=51 b=21 result=72 a_ref=('res', 1) b_ref=('slot', 0) compound=False ambiguous=True
* aligned gold step: 2 (tier 1) (+, ('res', 1), ('slot', 0), 72)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 4 * 12 = 48; 1: 48 + 3 = 51; 2: 51 + 21 = 72; 3: 72 / 12 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4482 marked_p0 trace_correct=True
* step text: `Step 1: Total number of fruit trees = 18 streets / 2 = 9 fruit trees`
* parsed: op=/ a=18 b=2 result=9 a_ref=('const', Fraction(18, 1)) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (/, ('const', Fraction(18, 1)), ('const', Fraction(2, 1)), 9)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 18 / 2 = 9; 1: 9 / 3 = 3
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5599 terse_p0 trace_correct=False
* step text: `* Total distance = 40 miles + 24 miles + 4.8 miles = 68.8 miles`
* parsed: op=None a=None b=None result=344/5 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 3/5 * 40 = 24; 1: 24 + 40 = 64; 2: 5 * 24 = 120; 3: 120 + 64 = 184
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4274 terse_p0 trace_correct=True
* step text: `* $2000 - $800 = $1200`
* parsed: op=- a=2000 b=800 result=1200 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('slot', 0), ('res', 0), 1200)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2/5 * 2000 = 800; 1: 2000 - 800 = 1200; 2: 2/5 * 1200 = 480; 3: 1200 - 480 = 720; 4: 2/5 * 720 = 288; 5: 720 - 288 = 432
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5895 marked_p0 trace_correct=True
* step text: `Step 3: James jumps 2/3 the height of Jacob, so James can jump 24 inches * (2/3) = 16 inches.`
* parsed: op=* a=24 b=2/3 result=16 a_ref=('res', 1) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (*, ('res', 1), ('slot', 1), 16)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 6 * 2 = 12; 1: 12 * 2 = 24; 2: 24 * 2/3 = 16
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4983 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total water consumption per hour.
800 ml/hour + 400 ml/hour = 1200 ml/hour`
* parsed: op=+ a=800 b=400 result=1200 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (+, ('slot', 0), ('slot', 1), 1200)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 800 + 400 = 1200; 1: 1200 / 1000 = 6/5; 2: 24/5 / 6/5 = 4
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5566 marked_p0 trace_correct=False
* step text: `Step 4: Calculate the total distance traveled.
Total Distance = 5 + 7 + 5 = 17 blocks`
* parsed: op=None a=None b=None result=17 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 5 + 7 = 12; 1: 12 * 2 = 24
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3008 terse_p0 trace_correct=True
* step text: `* **Find the length of the head:** The head is one-tenth the length of the snake, so it's 10 feet / 10 = 1 foot long.`
* parsed: op=/ a=10 b=10 result=1 a_ref=('slot', 0) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('const', Fraction(1, 10)), ('slot', 0), 1)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 1/10 * 10 = 1; 1: 10 - 1 = 9
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-374 terse_p0 trace_correct=True
* step text: `* **Today's buyers:** There are 40 more buyers today than yesterday, so today has 25 + 40 = 65 buyers.`
* parsed: op=+ a=25 b=40 result=65 a_ref=('res', 0) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('slot', 0), ('res', 0), 65)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 1/2 * 50 = 25; 1: 40 + 25 = 65; 2: 65 + 25 = 90; 3: 90 + 50 = 140
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-331 terse_p0 trace_correct=False
* step text: `* Asphalt needed for the entire road: 11 miles * 3 truckloads/mile = 33 truckloads`
* parsed: op=* a=11 b=3 result=33 a_ref=('rres', 6) b_ref=('const', Fraction(3, 1)) compound=False ambiguous=False
* aligned gold step: 4 (tier 4) (*, ('const', Fraction(3, 1)), ('res', 3), 15)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 4 * 2 = 8; 1: 8 - 1 = 7; 2: 16 - 7 = 9; 3: 9 - 4 = 5; 4: 3 * 5 = 15; 5: 15 * 2 = 30; 6: 30 / 5 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4166 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total value of Grace's coins.
   $1.00 + $0.50 = $1.50`
* parsed: op=+ a=1 b=1/2 result=3/2 a_ref=('rres', 0) b_ref=('rres', 1) compound=False ambiguous=False
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=False  rederived=False
* gold program: 0: 10 * 10 = 100; 1: 10 * 5 = 50; 2: 100 + 50 = 150
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5221 terse_p0 trace_correct=True
* step text: `* **Eggs left:** The chef had 60 eggs - 10 eggs = 50 eggs left.`
* parsed: op=- a=60 b=10 result=50 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (-, ('slot', 0), ('slot', 1), 50)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 60 - 10 = 50; 1: 50 / 5 = 10
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3648 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total time Billy spends on laundry.
Carla: Billy does 2 loads * 9 minutes/load = 18 minutes`
* parsed: op=* a=2 b=9 result=18 a_ref=('slot', 1) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 1), ('slot', 2), 18)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2 * 9 = 18; 1: 3 * 10 = 30; 2: 30 - 18 = 12; 3: 12 / 2 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4328 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total earnings.
Total Earnings = $24 + $28 = $52`
* parsed: op=+ a=24 b=28 result=52 a_ref=('res', 0) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (+, ('res', 0), ('res', 1), 52)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 3 * 8 = 24; 1: 2 * 14 = 28; 2: 24 + 28 = 52
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4809 marked_p0 trace_correct=False
* step text: `Step 6: Calculate the new balance after the second month.
The new balance is the original loan plus the interest: $202 + $2.02 = $204.02.`
* parsed: op=+ a=202 b=101/50 result=10201/50 a_ref=('rres', 2) b_ref=('rres', 4) compound=False ambiguous=False
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=False  rederived=False
* gold program: 0: 200 * 1/10 = 20; 1: 200 + 20 = 220; 2: 220 * 1/10 = 22
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5776 terse_p0 trace_correct=True
* step text: `* Number of applications: 16 hours / 2 hours/application = 8 applications`
* parsed: op=/ a=16 b=2 result=8 a_ref=('slot', 4) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (/, ('slot', 4), ('slot', 0), 8)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 16 / 2 = 8; 1: 3 * 8 = 24; 2: 24 / 12 = 2; 3: 2 * 7/2 = 7
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1762 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total number of cars with valid tickets and permanent parking passes.
    225 cars + 45 cars = 270 cars`
* parsed: op=+ a=225 b=45 result=270 a_ref=('res', 0) b_ref=('res', 2) compound=False ambiguous=False
* aligned gold step: 3 (tier 5) (-, ('slot', 2), ('res', 0), 75)
* error type: planning  extra_valid=True  rederived=False
* gold program: 0: 3/4 * 300 = 225; 1: 3/4 / 5 = 3/20; 2: 3/20 * 300 = 45; 3: 300 - 225 = 75; 4: 75 - 45 = 30
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6131 marked_p0 trace_correct=False
* step text: `Step 8: Calculate the total number of passengers: 24 + 180 + 96 + 90 = 400 passengers`
* parsed: op=None a=None b=None result=400 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: arithmetic  extra_valid=False  rederived=False
* gold program: 0: 12 * 2 = 24; 1: 2 * 15 = 30; 2: 2 * 2 = 4; 3: 4 * 2 = 8; 4: 30 * 3 = 90; 5: 52 - 12 = 40; 6: 40 - 2 = 38; 7: 38 - 4 = 34; 8: 34 - 30 = 4; 9: 4 * 1 = 4; 10: 24 + 30 = 54; 11: 54 + 8 = 62; 12: 62 + 90 = 152; 13: 152 + 4 = 156
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1963 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the amount saved.
Savings = $2880 - $1728 = $1152`
* parsed: op=- a=2880 b=1728 result=1152 a_ref=('res', 0) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (-, ('res', 0), ('res', 1), 1152)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 3/25 * 24000 = 2880; 1: 3/5 * 2880 = 1728; 2: 2880 - 1728 = 1152
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2621 terse_p0 trace_correct=True
* step text: `* **Total hangers:**  She has 7 + 4 + 3 + 2 = 16 colored hangers.`
* parsed: op=None a=None b=None result=16 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 4 (tier 3) (+, ('res', 3), ('res', 1), 16)
* error type: None  extra_valid=False  rederived=False
* gold program: 0: 4 - 1 = 3; 1: 3 - 1 = 2; 2: 7 + 4 = 11; 3: 11 + 3 = 14; 4: 14 + 2 = 16
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2578 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total number of rubber bands they have.
   They have 15 + 9 = 24 rubber bands together.`
* parsed: op=+ a=15 b=9 result=24 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('slot', 0), ('res', 0), 24)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 15 - 6 = 9; 1: 15 + 9 = 24
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-322 marked_p0 trace_correct=False
* step text: `Step 4: Calculate the amount of gratuity.
Gratuity = $126 - $90 = $36`
* parsed: op=- a=126 b=90 result=36 a_ref=('rres', 1) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 2 (tier 5) (*, ('res', 1), ('slot', 1), 9)
* error type: planning  extra_valid=False  rederived=False
* gold program: 0: 80 + 10 = 90; 1: 80 + 10 = 90; 2: 90 * 1/10 = 9; 3: 90 + 9 = 99; 4: 140 - 99 = 41
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1961 terse_p0 trace_correct=True
* step text: `* 4000 candies / 500 candies/day = 8 days`
* parsed: op=/ a=4000 b=500 result=8 a_ref=('slot', 1) b_ref=('rres', 2) compound=False ambiguous=False
* aligned gold step: 1 (tier 3) (/, ('res', 0), ('slot', 2), 8)
* error type: none  extra_valid=False  rederived=True
* gold program: 0: 4000 / 50 = 80; 1: 80 / 10 = 8
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3671 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the time Diana takes to bike the remaining 4 miles at 1 mph.
Time = Distance / Speed = 4 miles / 1 mph = 4 hours`
* parsed: op=/ a=4 b=1 result=4 a_ref=('res', 1) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (*, ('res', 1), ('slot', 2), 4)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2 * 3 = 6; 1: 10 - 6 = 4; 2: 4 * 1 = 4; 3: 2 + 4 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1827 terse_p0 trace_correct=True
* step text: `* Total: 13 eggs/day * 5 days/week = 65 eggs`
* parsed: op=* a=13 b=5 result=65 a_ref=('res', 1) b_ref=('const', Fraction(5, 1)) compound=False ambiguous=False
* aligned gold step: 2 (tier 4) (*, ('res', 0), ('const', Fraction(52, 1)), 260)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 3 + 2 = 5; 1: 5 + 8 = 13; 2: 5 * 52 = 260; 3: 13 * 260 = 3380
* manual check: [ ] correct  [ ] wrong (note):
