# Random parsed steps (S1c traces) for manual checking

## gsm8k-train-2590 marked_p0 trace_correct=False
* step text: `Step 4: Calculate the total number of glasses in the tall and wide cupboards.
The tall cupboard holds 20 glasses, and the wide cupboard holds 40 glasses, for a total of 20 glasses + 40 glasses = 60 gl`
* parsed: op=+ a=20 b=40 result=60 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 3 (tier 1) (+, ('slot', 0), ('res', 0), 60)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 20 * 2 = 40; 1: 15 / 3 = 5; 2: 15 - 5 = 10; 3: 20 + 40 = 60; 4: 60 + 10 = 70
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6490 marked_p0 trace_correct=True
* step text: `Step 3: Subtract the withdrawal: $80 - $4 = $76`
* parsed: op=- a=80 b=4 result=76 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('res', 0), ('slot', 2), 76)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 65 + 15 = 80; 1: 80 - 4 = 76
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5819 marked_p0 trace_correct=True
* step text: `Step 4: Calculate the total water needed by the ducks.
   Total water for ducks = 20 ducks * 0.25 gallons/duck = 5 gallons`
* parsed: op=* a=20 b=1/4 result=5 a_ref=('slot', 5) b_ref=('const', Fraction(1, 4)) compound=False ambiguous=False
* aligned gold step: 3 (tier 1) (*, ('slot', 5), ('const', Fraction(1, 4)), 5)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 15 * 4 = 60; 1: 60 * 1/2 = 30; 2: 10 * 4 = 40; 3: 20 * 1/4 = 5; 4: 5 + 40 = 45; 5: 45 + 30 = 75; 6: 75 / 3 = 25
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1326 terse_p0 trace_correct=False
* step text: `* **Calculate the original price per tire:** $264 / 4 tires = $66/tire`
* parsed: op=/ a=264 b=4 result=66 a_ref=('rres', 2) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 4) (/, ('slot', 2), ('slot', 0), 9)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 36 / 4 = 9; 1: 75 + 9 = 84
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4161 terse_p0 trace_correct=True
* step text: `* Mr. Wells cuts 60% of the flowers, so he cuts 20,000 * 0.60 = 12,000 flowers`
* parsed: op=* a=20000 b=3/5 result=12000 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('slot', 2), ('res', 0), 12000)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 50 * 400 = 20000; 1: 3/5 * 20000 = 12000; 2: 20000 - 12000 = 8000
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6648 marked_p0 trace_correct=True
* step text: `Step 1: Cindi bought pencils at $0.50 each, so we need to find out how many pencils she bought. 
Answer: Cindi bought 30 / 0.50 = 60 pencils.`
* parsed: op=/ a=30 b=1/2 result=60 a_ref=('slot', 1) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (/, ('slot', 1), ('slot', 2), 60)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 30 / 1/2 = 60; 1: 2 * 60 = 120; 2: 3 * 120 = 360; 3: 120 + 360 = 480
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5221 marked_p0 trace_correct=True
* step text: `Step 2: Divide the remaining eggs by the number of eggs per cake to find the number of cakes. 50 eggs / 5 eggs/cake = 10 cakes`
* parsed: op=/ a=50 b=5 result=10 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (/, ('res', 0), ('slot', 2), 10)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 60 - 10 = 50; 1: 50 / 5 = 10
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6860 terse_p0 trace_correct=True
* step text: `* **Weight loss over a year:** Calvin loses 8 pounds/month * 12 months = 96 pounds`
* parsed: op=* a=8 b=12 result=96 a_ref=('slot', 1) b_ref=('const', Fraction(12, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('const', Fraction(12, 1)), ('slot', 1), 96)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 12 * 8 = 96; 1: 250 - 96 = 154
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6243 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total cost before the tip.
$50 + $6 = $56`
* parsed: op=+ a=50 b=6 result=56 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('slot', 0), ('res', 0), 56)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 50 * 3/25 = 6; 1: 50 + 6 = 56; 2: 56 + 5 = 61
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-674 terse_p0 trace_correct=True
* step text: `* Total full spots: 400 - 214 = 186 full parking spots`
* parsed: op=- a=400 b=214 result=186 a_ref=('res', 0) b_ref=('res', 5) compound=False ambiguous=False
* aligned gold step: 6 (tier 1) (-, ('res', 0), ('res', 5), 186)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 4 * 100 = 400; 1: 58 + 2 = 60; 2: 60 + 5 = 65; 3: 58 + 60 = 118; 4: 118 + 65 = 183; 5: 183 + 31 = 214; 6: 400 - 214 = 186
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6648 terse_p0 trace_correct=True
* step text: `* Donna and Marcia bought a total of 360 + 120 = 480 pencils.`
* parsed: op=+ a=360 b=120 result=480 a_ref=('res', 2) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 3 (tier 1) (+, ('res', 1), ('res', 2), 480)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 30 / 1/2 = 60; 1: 2 * 60 = 120; 2: 3 * 120 = 360; 3: 120 + 360 = 480
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-117 terse_p0 trace_correct=False
* step text: `* **Clownfish in display tank:**  There are now 12 - 4 = 8 clownfish in the display tank.`
* parsed: op=- a=12 b=4 result=8 a_ref=('res', 2) b_ref=('res', 2) compound=False ambiguous=False
* aligned gold step: 2 (tier 3) (/, ('res', 1), ('const', Fraction(3, 1)), 8)
* error type: planning  extra_valid=False  rederived=False
* gold program: 0: 100 / 2 = 50; 1: 50 - 26 = 24; 2: 24 / 3 = 8; 3: 24 - 8 = 16
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5186 marked_p0 trace_correct=False
* step text: `Step 3:  Janet's age 8 years ago was 21 / 2 = 10.5 years old.`
* parsed: op=/ a=21 b=2 result=21/2 a_ref=('res', 1) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (/, ('res', 1), ('const', Fraction(2, 1)), 7)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 37 - 15 = 22; 1: 22 - 8 = 14; 2: 14 / 2 = 7; 3: 7 + 8 = 15; 4: 15 - 3 = 12; 5: 12 + 5 = 17
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2764 terse_p0 trace_correct=False
* step text: `* After Brian's purchase, there were 100 + 29 + 9 - 74 = 66 animals left.`
* parsed: op=None a=None b=None result=66 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: arithmetic  extra_valid=False  rederived=False
* gold program: 0: 100 + 29 = 129; 1: 129 + 9 = 138; 2: 138 / 2 = 69; 3: 69 + 37 = 106; 4: 106 / 2 = 53
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5876 terse_p0 trace_correct=True
* step text: `* Fifth stack height: 10 * 2 = 20 blocks`
* parsed: op=* a=10 b=2 result=20 a_ref=('res', 0) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=True
* aligned gold step: 3 (tier 1) (*, ('res', 0), ('const', Fraction(2, 1)), 20)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 7 + 3 = 10; 1: 10 - 6 = 4; 2: 4 + 10 = 14; 3: 10 * 2 = 20; 4: 7 + 10 = 17; 5: 17 + 4 = 21; 6: 21 + 14 = 35; 7: 35 + 20 = 55
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2551 terse_p0 trace_correct=False
* step text: `* **Points scored in normal games:** Duke scored 2 * 5 = 10 points in normal games.`
* parsed: op=* a=2 b=5 result=10 a_ref=('slot', 5) b_ref=('slot', 3) compound=False ambiguous=False
* aligned gold step: 1 (tier 4) (*, ('slot', 1), ('const', Fraction(1, 1)), 5)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 17 + 5 = 22; 1: 5 * 1 = 5; 2: 4 * 2 = 8; 3: 22 - 5 = 17; 4: 17 - 8 = 9; 5: 9 / 3 = 3; 6: 3 - 2 = 1
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2101 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the total number of slices needed. 
   Total slices = 10 people * 2 slices/person = 20 slices`
* parsed: op=* a=10 b=2 result=20 a_ref=('const', Fraction(10, 1)) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('res', 0), ('slot', 0), 20)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 1 + 9 = 10; 1: 10 * 2 = 20; 2: 20 / 4 = 5
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5186 terse_p0 trace_correct=True
* step text: `* 8 years ago, James was twice Janet's age, meaning James was 22 - 8 = 14 years old.`
* parsed: op=- a=22 b=8 result=14 a_ref=('res', 0) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('res', 0), ('slot', 0), 14)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 37 - 15 = 22; 1: 22 - 8 = 14; 2: 14 / 2 = 7; 3: 7 + 8 = 15; 4: 15 - 3 = 12; 5: 12 + 5 = 17
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5788 terse_p0 trace_correct=True
* step text: `* Time = 8 kilometers / 10 minutes/kilometer = 0.8 hours`
* parsed: op=/ a=8 b=10 result=4/5 a_ref=('slot', 3) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 5) (*, ('const', Fraction(6, 1)), ('slot', 3), 48)
* error type: planning  extra_valid=True  rederived=False
* gold program: 0: 7/2 * 60 = 210; 1: 210 / 21 = 10; 2: 6 * 8 = 48
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5876 marked_p0 trace_correct=True
* step text: `Step 4: Calculate the number of blocks in the fifth stack.
Fifth stack height = Second stack height * 2 = 10 blocks * 2 = 20 blocks`
* parsed: op=* a=10 b=2 result=20 a_ref=('res', 0) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=True
* aligned gold step: 3 (tier 1) (*, ('res', 0), ('const', Fraction(2, 1)), 20)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 7 + 3 = 10; 1: 10 - 6 = 4; 2: 4 + 10 = 14; 3: 10 * 2 = 20; 4: 7 + 10 = 17; 5: 17 + 4 = 21; 6: 21 + 14 = 35; 7: 35 + 20 = 55
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5246 marked_p0 trace_correct=False
* step text: `Step 2: Calculate the cost for walking two dogs for 7 minutes.
Cost = $20 + ($1/minute * 7 minutes) = $20 + $7 = $27`
* parsed: op=None a=None b=None result=27 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 3 (tier 3) (+, ('slot', 0), ('res', 2), 27)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 10 * 1 = 10; 1: 20 + 10 = 30; 2: 7 * 1 = 7; 3: 20 + 7 = 27; 4: 2 * 27 = 54; 5: 9 * 1 = 9; 6: 20 + 9 = 29; 7: 3 * 29 = 87; 8: 30 + 54 = 84; 9: 84 + 87 = 171
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4458 marked_p0 trace_correct=False
* step text: `Step 1: Calculate the hours open Monday through Friday.
    Jean's business is open from 4 pm to 10 pm, Monday through Friday.
    Hours open: 10 pm - 4 pm = 6 hours`
* parsed: op=- a=10 b=4 result=6 a_ref=('slot', 2) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (-, ('slot', 2), ('slot', 0), 6)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 10 - 4 = 6; 1: 6 * 5 = 30; 2: 10 - 6 = 4; 3: 4 * 2 = 8; 4: 30 + 8 = 38
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-7006 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the total yards of floss needed. 
   Total yards = 20 students * 1.5 yards/student = 30 yards`
* parsed: op=* a=20 b=3/2 result=30 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('slot', 1), 30)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 20 * 3/2 = 30; 1: 35 - 30 = 5
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1551 terse_p0 trace_correct=True
* step text: `* She gives 25 - 1 = 24 time-outs for throwing food.`
* parsed: op=- a=25 b=1 result=24 a_ref=('res', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('res', 0), ('slot', 1), 24)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 5 * 5 = 25; 1: 25 - 1 = 24; 2: 24 / 3 = 8; 3: 8 + 24 = 32; 4: 32 + 5 = 37; 5: 37 * 5 = 185
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2540 marked_p0 trace_correct=False
* step text: `Step 6:  In three years, the art piece will be worth 3 * $1998.5 = $5995.5`
* parsed: op=* a=3 b=3997/2 result=11991/2 a_ref=('const', Fraction(3, 1)) b_ref=('const', Fraction(3997, 2)) compound=False ambiguous=False
* aligned gold step: 0 (tier 4) (*, ('const', Fraction(3, 1)), ('slot', 0), 12000)
* error type: binding  extra_valid=False  rederived=False
* gold program: 0: 3 * 4000 = 12000; 1: 12000 - 4000 = 8000
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6933 terse_p0 trace_correct=True
* step text: `* 500 pounds (total mince) - 200 pounds (lasagna mince) = 300 pounds of mince`
* parsed: op=- a=500 b=200 result=300 a_ref=('slot', 3) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('slot', 3), ('res', 0), 300)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 100 * 2 = 200; 1: 500 - 200 = 300; 2: 300 / 3 = 100
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1559 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total cost of the blue pens.
Total cost of blue pens = 10 pens * 10 cents/pen = 100 cents`
* parsed: op=* a=10 b=10 result=100 a_ref=('const', Fraction(10, 1)) b_ref=('const', Fraction(10, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 4) (*, ('const', Fraction(10, 1)), ('const', Fraction(1, 10)), 1)
* error type: binding  extra_valid=True  rederived=False
* gold program: 0: 10 * 1/10 = 1; 1: 2 * 1/10 = 1/5; 2: 15 * 1/5 = 3; 3: 3 + 1 = 4
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5659 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the number of visitors in November.
100 * 1.15 = 115 visitors`
* parsed: op=* a=100 b=23/20 result=115 a_ref=('slot', 0) b_ref=('const', Fraction(23, 20)) compound=False ambiguous=False
* aligned gold step: 1 (tier 3) (+, ('slot', 0), ('res', 0), 115)
* error type: none  extra_valid=False  rederived=True
* gold program: 0: 100 * 3/20 = 15; 1: 100 + 15 = 115; 2: 115 + 15 = 130; 3: 100 + 115 = 215; 4: 215 + 130 = 345
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4376 terse_p0 trace_correct=True
* step text: `* Mondays: $5/hour * 2 hours = $10`
* parsed: op=* a=5 b=2 result=10 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 4) (*, ('slot', 0), ('res', 1), 30)
* error type: binding  extra_valid=True  rederived=False
* gold program: 0: 2 + 1 = 3; 1: 3 + 3 = 6; 2: 5 * 6 = 30; 3: 180 / 30 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-132 terse_p0 trace_correct=True
* step text: `* **Calculate the total weight:** 5 packs * 4 pounds/pack = 20 pounds`
* parsed: op=* a=5 b=4 result=20 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('slot', 1), 20)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 5 * 4 = 20; 1: 20 * 11/2 = 110
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-7395 marked_p0 trace_correct=True
* step text: `Step 1: Matt has ten times as many jellybeans as Steve.
Matt's jellybeans = 10 * 84 = 840`
* parsed: op=* a=10 b=84 result=840 a_ref=('const', Fraction(10, 1)) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('const', Fraction(10, 1)), 840)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 84 * 10 = 840; 1: 840 / 2 = 420
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-674 marked_p0 trace_correct=True
* step text: `Step 4: Calculate the total number of parking spots: 4 levels * 100 spots/level = 400 spots`
* parsed: op=* a=4 b=100 result=400 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('slot', 1), 400)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 4 * 100 = 400; 1: 58 + 2 = 60; 2: 60 + 5 = 65; 3: 58 + 60 = 118; 4: 118 + 65 = 183; 5: 183 + 31 = 214; 6: 400 - 214 = 186
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1594 terse_p0 trace_correct=True
* step text: `* **Oranges remaining:** They started with 90 oranges and ate 60, leaving 90 - 60 = 30 oranges.`
* parsed: op=- a=90 b=60 result=30 a_ref=('slot', 1) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (-, ('slot', 1), ('res', 1), 30)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2 * 20 = 40; 1: 20 + 40 = 60; 2: 90 - 60 = 30
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6420 terse_p0 trace_correct=False
* step text: `* **Step 5:**  Total distance walked: 20 + 200 + 440 + 220 = 880 feet`
* parsed: op=None a=None b=None result=880 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=False  rederived=False
* gold program: 0: 20 + 200 = 220; 1: 220 * 2 = 440; 2: 220 + 440 = 660; 3: 660 / 2 = 330; 4: 660 + 330 = 990; 5: 990 * 2 = 1980
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-408 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total weight of the two children: 2 children * 64 pounds/child = 128 pounds`
* parsed: op=* a=2 b=64 result=128 a_ref=('const', Fraction(2, 1)) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('slot', 1), ('const', Fraction(2, 1)), 128)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 140 * 3 = 420; 1: 64 * 2 = 128; 2: 420 + 128 = 548; 3: 600 - 548 = 52
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3435 marked_p0 trace_correct=False
* step text: `Step 1: Calculate the total calories from the burritos.
10 burritos * 120 calories/burrito = 1200 calories`
* parsed: op=* a=10 b=120 result=1200 a_ref=('slot', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('slot', 2), 1200)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 10 * 120 = 1200; 1: 1200 / 6 = 200; 2: 5 * 400 = 2000; 3: 2000 / 8 = 250; 4: 250 - 200 = 50
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-333 terse_p0 trace_correct=True
* step text: `* **Total chocolates:** 12 + 10 = 22 chocolates.`
* parsed: op=+ a=12 b=10 result=22 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 4 (tier 3) (+, ('res', 3), ('res', 1), 22)
* error type: none  extra_valid=False  rederived=True
* gold program: 0: 1 * 4 = 4; 1: 2 * 4 = 8; 2: 1 * 4 = 4; 3: 10 + 4 = 14; 4: 14 + 8 = 22
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2933 marked_p0 trace_correct=True
* step text: `Step 4: Subtract the books taken out on Friday: 64 - 35 = 29`
* parsed: op=- a=64 b=35 result=29 a_ref=('res', 1) b_ref=('slot', 3) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (-, ('res', 1), ('slot', 3), 29)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 235 - 227 = 8; 1: 8 + 56 = 64; 2: 64 - 35 = 29
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5201 marked_p0 trace_correct=False
* step text: `Step 3: Calculate the total time Kira takes to travel between the first and third station.
Total time = 120 minutes + 30 minutes = 150 minutes`
* parsed: op=+ a=120 b=30 result=150 a_ref=('res', 1) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (+, ('res', 1), ('slot', 1), 270)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 2 + 2 = 4; 1: 4 * 60 = 240; 2: 240 + 30 = 270
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6536 terse_p0 trace_correct=True
* step text: `* **Total attendance this week:** 80 people + 60 people + 95 people = 235 people`
* parsed: op=None a=None b=None result=235 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 3 (tier 3) (+, ('res', 2), ('res', 1), 235)
* error type: none  extra_valid=False  rederived=False
* gold program: 0: 80 - 20 = 60; 1: 80 + 15 = 95; 2: 80 + 60 = 140; 3: 140 + 95 = 235; 4: 235 - 200 = 35
* manual check: [ ] correct  [ ] wrong (note):
