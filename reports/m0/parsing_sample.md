# Random parsed steps (S1c traces) for manual checking

## gsm8k-train-4870 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the revenue from children's tickets.
The number of children's tickets sold is 28.
The revenue from children's tickets is 28 * $13 = $364.`
* parsed: op=* a=28 b=13 result=364 a_ref=('slot', 5) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 2 (*, ('slot', 5), ('res', 1), 364)
* error type: binding
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5252 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the number of parents who rated Excellent. 
15% of 120 = (15/100) * 120 = 18 parents`
* parsed: op=None a=None b=None result=18 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None 
* error type: None
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-606 marked_p0 trace_correct=False
* step text: `Step 2: Calculate the number of yellow marbles.
   Yellow marbles make up 20% of the green marbles, so there are 60 * 0.20 = 12 yellow marbles.`
* parsed: op=* a=60 b=1/5 result=12 a_ref=('res', 0) b_ref=('const', Fraction(1, 5)) compound=False ambiguous=False
* aligned gold step: 1 (*, ('res', 0), ('slot', 1), 1200)
* error type: binding
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3005 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total amount spent.
Together they spent $400 + $560 = $960.`
* parsed: op=+ a=400 b=560 result=960 a_ref=('slot', 2) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (+, ('slot', 2), ('res', 0), 960)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6490 marked_p0 trace_correct=True
* step text: `Step 2: Add the deposit: $65 + $15 = $80`
* parsed: op=+ a=65 b=15 result=80 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (+, ('slot', 0), ('slot', 1), 80)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6074 marked_p0 trace_correct=True
* step text: `Step 3: He has 500 - 250 = 250 Legos left.`
* parsed: op=- a=500 b=250 result=250 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (-, ('res', 0), ('slot', 1), 245)
* error type: binding
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5083 marked_p0 trace_correct=True
* step text: `Step 3: Determine the number of bids for each person.
Since the price increased by $5 each time, and there were 10 bids in total, each person bid 10 / 2 = 5 times.`
* parsed: op=/ a=10 b=2 result=5 a_ref=('res', 1) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 2 (/, ('res', 1), ('const', Fraction(2, 1)), 5)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3691 terse_p0 trace_correct=True
* step text: `* $90 + $96 = $186`
* parsed: op=+ a=90 b=96 result=186 a_ref=('const', Fraction(90, 1)) b_ref=('const', Fraction(96, 1)) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5885 terse_p0 trace_correct=True
* step text: `* Total t-shirts made: 5 + 10 = 15 t-shirts`
* parsed: op=+ a=5 b=10 result=15 a_ref=('res', 0) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (+, ('res', 0), ('res', 1), 15)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4459 terse_p0 trace_correct=True
* step text: `* $4.00 - $3.70 = $0.30`
* parsed: op=- a=4 b=37/10 result=3/10 a_ref=('slot', 0) b_ref=('rres', 6) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-7306 terse_p0 trace_correct=True
* step text: `* Cost of 2 sandwiches = 2 * $2.50 = $5`
* parsed: op=* a=2 b=5/2 result=5 a_ref=('slot', 0) b_ref=('slot', 3) compound=False ambiguous=True
* aligned gold step: 1 (*, ('slot', 3), ('slot', 2), 5)
* error type: binding
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2484 marked_p0 trace_correct=True
* step text: `Step 3: Total people left relaxing = 21 + 15 + 18 = 54`
* parsed: op=None a=None b=None result=54 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None 
* error type: None
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6384 marked_p0 trace_correct=True
* step text: `Step 5: Find Kendra's age in 3 years.
Kendra is currently 18 years old, so in 3 years she will be 18 + 3 = 21 years old.`
* parsed: op=+ a=18 b=3 result=21 a_ref=('slot', 1) b_ref=('res', 1) compound=False ambiguous=True
* aligned gold step: 4 (+, ('slot', 1), ('res', 1), 21)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1551 terse_p0 trace_correct=True
* step text: `* **Food throwing time-outs:** Marcy gives 5 * 5 = 25 time-outs for throwing food.`
* parsed: op=* a=5 b=5 result=25 a_ref=('slot', 0) b_ref=('slot', 0) compound=False ambiguous=True
* aligned gold step: 0 (*, ('slot', 0), ('slot', 0), 25)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3445 marked_p0 trace_correct=False
* step text: `Step 2: Calculate the cost of a single kid's meal.
$5.00 / 3 = $1.67`
* parsed: op=/ a=5 b=3 result=167/100 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=True
* aligned gold step: 0 (+, ('slot', 0), ('slot', 1), 8)
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1551 terse_p0 trace_correct=True
* step text: `* She gives 25 - 1 = 24 time-outs for throwing food.`
* parsed: op=- a=25 b=1 result=24 a_ref=('res', 0) b_ref=('slot', 1) compound=False ambiguous=True
* aligned gold step: 1 (-, ('res', 0), ('slot', 1), 24)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1144 terse_p0 trace_correct=False
* step text: `* The older brother's age is 3 * 5 = 15 years old.`
* parsed: op=* a=3 b=5 result=15 a_ref=('const', Fraction(3, 1)) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (*, ('const', Fraction(3, 1)), ('slot', 1), 15)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2764 marked_p0 trace_correct=True
* step text: `Step 6: Calculate the number of male animals in the barn.
Since exactly half of the animals are male, there are 106 / 2 = 53 male animals.`
* parsed: op=/ a=106 b=2 result=53 a_ref=('rres', 4) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6739 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the number of candles Ethan makes. 
Ethan makes 10 - 3 = 7 candles.`
* parsed: op=- a=10 b=3 result=7 a_ref=('slot', 2) b_ref=('const', Fraction(3, 1)) compound=False ambiguous=False
* aligned gold step: 1 (-, ('slot', 2), ('const', Fraction(3, 1)), 7)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1560 terse_p0 trace_correct=True
* step text: `* Budget - price of smaller frame = $60 - $54 = $6`
* parsed: op=- a=60 b=54 result=6 a_ref=('slot', 1) b_ref=('rres', 6) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3896 terse_p0 trace_correct=True
* step text: `* **Files missing:**  She has 60 - 45 = 15 files missing.`
* parsed: op=- a=60 b=45 result=15 a_ref=('slot', 0) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (-, ('slot', 0), ('res', 1), 15)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1203 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the total amount spent. 
$1.75 + $1.25 = $3.00`
* parsed: op=+ a=7/4 b=5/4 result=3 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (+, ('slot', 0), ('slot', 1), 3)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1019 marked_p0 trace_correct=False
* step text: `Step 7: Calculate the number of cookies thrown on the floor: 93 - 81 = 12.`
* parsed: op=- a=93 b=81 result=12 a_ref=('slot', 4) b_ref=('rres', 4) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4008 terse_p0 trace_correct=True
* step text: `* **Calculate the number of days Kris was suspended:** Kris was suspended for three times the number of fingers and toes, so she was suspended for 3 * 20 = 60 days.`
* parsed: op=* a=3 b=20 result=60 a_ref=('slot', 0) b_ref=('const', Fraction(20, 1)) compound=False ambiguous=False
* aligned gold step: 0 (*, ('slot', 0), ('const', Fraction(20, 1)), 60)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5876 terse_p0 trace_correct=True
* step text: `* Second stack height: 7 + 3 = 10 blocks`
* parsed: op=+ a=7 b=3 result=10 a_ref=('slot', 1) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 0 (+, ('slot', 1), ('slot', 2), 10)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6967 terse_p0 trace_correct=True
* step text: `* Twice the number of Reeta's pencils is 2 * 20 = 40 pencils.`
* parsed: op=* a=2 b=20 result=40 a_ref=('const', Fraction(2, 1)) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (+, ('const', Fraction(44, 1)), ('slot', 1), 64)
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1224 marked_p0 trace_correct=True
* step text: `Step 1: Calculate the growth of the initial investment. 
   <calculation> = $80 * 0.15 = $12</result>`
* parsed: op=* a=80 b=3/20 result=12 a_ref=('slot', 0) b_ref=('const', Fraction(3, 20)) compound=False ambiguous=False
* aligned gold step: 0 (*, ('slot', 0), ('const', Fraction(3, 20)), 12)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4406 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total number of petals Collin has.
Each flower has 4 petals, and Collin has 36 flowers, so he has 36 * 4 = 144 petals.`
* parsed: op=* a=36 b=4 result=144 a_ref=('res', 1) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 2 (*, ('res', 1), ('slot', 2), 144)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5387 terse_p0 trace_correct=False
* step text: `* **Total emails:** 600 + 750 = 1350 emails`
* parsed: op=+ a=600 b=750 result=1350 a_ref=('const', Fraction(600, 1)) b_ref=('const', Fraction(750, 1)) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3904 terse_p0 trace_correct=True
* step text: `* Cost of paints: $80 - $75 = $5`
* parsed: op=- a=80 b=75 result=5 a_ref=('slot', 8) b_ref=('rres', 15) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2308 terse_p0 trace_correct=False
* step text: `* **Calculate the number of bandages before:** Two dozen has 2 * 12 = 24 bandages. 8 less than two dozen is 24 - 8 = 16 bandages.`
* parsed: op=* a=2 b=12 result=24 a_ref=('const', Fraction(2, 1)) b_ref=('const', Fraction(12, 1)) compound=False ambiguous=False
* aligned gold step: 2 (+, ('const', Fraction(2, 1)), ('const', Fraction(3, 1)), 5)
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6867 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total cost of the socks and shoes: $19.00 + $92.00 = $111.00`
* parsed: op=+ a=19 b=92 result=111 a_ref=('const', Fraction(19, 1)) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5473 terse_p0 trace_correct=True
* step text: `* Days worked: 30 / 2 = 15 days`
* parsed: op=/ a=30 b=2 result=15 a_ref=('slot', 1) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6559 terse_p0 trace_correct=False
* step text: `* **White tiles:** There are 20 - 7 = 13 white tiles.`
* parsed: op=- a=20 b=7 result=13 a_ref=('slot', 0) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (+, ('res', 1), ('const', Fraction(6, 1)), 13)
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3047 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the number of seats sold.
   Seats sold = 200 seats * (3/4) = 150 seats`
* parsed: op=/ a=3 b=4 result=150 a_ref=('slot', 3) b_ref=('slot', 4) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-948 marked_p0 trace_correct=True
* step text: `Step 1: Find how many times Jason goes per week.
Jason goes 4 times more often than William, who goes 2 times per week.
Jason goes 4 * 2 = 8 times per week.`
* parsed: op=* a=4 b=2 result=8 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=True
* aligned gold step: 0 (*, ('slot', 0), ('slot', 1), 8)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6847 marked_p0 trace_correct=True
* step text: `Step 5: Calculate the total amount of food both sides consume.
The total amount of food consumed is 28000 + 40000 = 68000 pounds per day.`
* parsed: op=+ a=28000 b=40000 result=68000 a_ref=('rres', 2) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 3 (*, ('res', 2), ('res', 1), 28000)
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-117 terse_p0 trace_correct=False
* step text: `* **Clownfish leaving display tank:** A third of the clownfish leave the display tank, so 12 / 3 = 4 clownfish leave.`
* parsed: op=/ a=12 b=3 result=4 a_ref=('res', 2) b_ref=('const', Fraction(3, 1)) compound=False ambiguous=False
* aligned gold step: 3 (-, ('res', 1), ('res', 2), 16)
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1144 marked_p0 trace_correct=False
* step text: `Step 3: His oldest brother is 1 year older than twice Michael's age when he was a year younger, so his age is 2m - 1 + 1 = 2m.`
* parsed: op=+ a=1 b=1 result=2 a_ref=('slot', 0) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: None 
* error type: planning
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4967 terse_p0 trace_correct=True
* step text: `* Total water: 90,000 + 120,000 = 210,000 gallons`
* parsed: op=+ a=90000 b=120000 result=210000 a_ref=('res', 3) b_ref=('res', 4) compound=False ambiguous=False
* aligned gold step: 5 (+, ('res', 4), ('res', 3), 210000)
* error type: none
* manual check: [ ] correct  [ ] wrong (note):
