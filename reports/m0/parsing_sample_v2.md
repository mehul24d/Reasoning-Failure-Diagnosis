# Random parsed steps (S1c traces) for manual checking

## gsm8k-train-1762 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total number of cars with valid tickets and permanent parking passes.
    225 cars + 45 cars = 270 cars`
* parsed: op=+ a=225 b=45 result=270 a_ref=('res', 0) b_ref=('res', 2) compound=False ambiguous=False
* aligned gold step: 3 (tier 5) (-, ('slot', 2), ('res', 0), 75)
* error type: planning  extra_valid=False
* gold program: 0: 3/4 * 300 = 225; 1: 3/4 / 5 = 3/20; 2: 3/20 * 300 = 45; 3: 300 - 225 = 75; 4: 75 - 45 = 30
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-831 terse_p0 trace_correct=True
* step text: `* **Sid's popsicles:** Sid has twice as many as Steve, who has 12. So Sid has 2 * 12 = 24 popsicle sticks.`
* parsed: op=* a=2 b=12 result=24 a_ref=('const', Fraction(2, 1)) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 0), ('const', Fraction(2, 1)), 24)
* error type: none  extra_valid=False
* gold program: 0: 12 * 2 = 24; 1: 24 * 3 = 72; 2: 24 + 72 = 96; 3: 96 + 12 = 108
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3623 marked_p0 trace_correct=True
* step text: `Step 5: Calculate the total time spent on parking, walking, and the metal detector in a week: 8 minutes/day * 5 days + 90 minutes = 40 minutes + 90 minutes = 130 minutes`
* parsed: op=+ a=5 b=90 result=40 a_ref=('slot', 0) b_ref=('const', Fraction(90, 1)) compound=False ambiguous=True
* aligned gold step: 1 (tier 3) (*, ('res', 0), ('slot', 0), 40)
* error type: planning  extra_valid=False
* gold program: 0: 5 + 3 = 8; 1: 8 * 5 = 40; 2: 2 * 30 = 60; 3: 3 * 10 = 30; 4: 30 + 60 = 90; 5: 90 + 40 = 130
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1540 terse_p0 trace_correct=False
* step text: `* Sam invested three times the original amount, so he has $17280 * 3 = $51840`
* parsed: op=* a=17280 b=3 result=51840 a_ref=('res', 2) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=True
* gold program: 0: 10000 * 6/5 = 12000; 1: 12000 * 6/5 = 14400; 2: 14400 * 6/5 = 17280; 3: 3 * 17280 = 51840; 4: 51840 * 23/20 = 59616
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6967 terse_p0 trace_correct=True
* step text: `* Anika has 4 more than twice Reeta's pencils, so she has 40 + 4 = 44 pencils.`
* parsed: op=+ a=40 b=4 result=44 a_ref=('res', 0) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('res', 0), ('slot', 0), 44)
* error type: none  extra_valid=False
* gold program: 0: 2 * 20 = 40; 1: 40 + 4 = 44; 2: 44 + 20 = 64
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6209 terse_p0 trace_correct=True
* step text: `* **Total:** Gilbert had 4 basil + 1 parsley = 5 herb plants when spring ended.`
* parsed: op=+ a=4 b=1 result=5 a_ref=('res', 0) b_ref=('const', Fraction(1, 1)) compound=False ambiguous=False
* aligned gold step: 3 (tier 3) (-, ('res', 2), ('const', Fraction(2, 1)), 5)
* error type: planning  extra_valid=False
* gold program: 0: 3 + 1 = 4; 1: 4 + 2 = 6; 2: 6 + 1 = 7; 3: 7 - 2 = 5
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6648 terse_p0 trace_correct=True
* step text: `* Donna and Marcia bought a total of 360 + 120 = 480 pencils.`
* parsed: op=+ a=360 b=120 result=480 a_ref=('res', 2) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 3 (tier 1) (+, ('res', 1), ('res', 2), 480)
* error type: none  extra_valid=False
* gold program: 0: 30 / 1/2 = 60; 1: 2 * 60 = 120; 2: 3 * 120 = 360; 3: 120 + 360 = 480
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5331 terse_p0 trace_correct=True
* step text: `* **Calculate the total cost:** The total cost of both items is the cost of the pencil plus the cost of the pen: $2 + $11 = $13.`
* parsed: op=+ a=2 b=11 result=13 a_ref=('slot', 1) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('res', 0), ('slot', 1), 13)
* error type: none  extra_valid=False
* gold program: 0: 9 + 2 = 11; 1: 11 + 2 = 13
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2764 marked_p0 trace_correct=True
* step text: `Step 6: Calculate the number of male animals in the barn.
Since exactly half of the animals are male, there are 106 / 2 = 53 male animals.`
* parsed: op=/ a=106 b=2 result=53 a_ref=('res', 4) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=False
* gold program: 0: 100 + 29 = 129; 1: 129 + 9 = 138; 2: 138 / 2 = 69; 3: 69 + 37 = 106; 4: 106 / 2 = 53
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1203 marked_p0 trace_correct=True
* step text: `Step 2: Add the amount spent to the amount left.
$3.00 + $6.00 = $9.00`
* parsed: op=+ a=3 b=6 result=9 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('slot', 2), ('res', 0), 9)
* error type: none  extra_valid=False
* gold program: 0: 7/4 + 5/4 = 3; 1: 6 + 3 = 9
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6847 terse_p0 trace_correct=True
* step text: `* Total: 40,000 pounds + 28,000 pounds = 68,000 pounds`
* parsed: op=+ a=40000 b=28000 result=68000 a_ref=('const', Fraction(40000, 1)) b_ref=('const', Fraction(28000, 1)) compound=False ambiguous=False
* aligned gold step: 4 (tier 1) (+, ('res', 3), ('res', 0), 68000)
* error type: none  extra_valid=False
* gold program: 0: 4000 * 10 = 40000; 1: 4000 - 500 = 3500; 2: 10 - 2 = 8; 3: 8 * 3500 = 28000; 4: 28000 + 40000 = 68000
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-374 terse_p0 trace_correct=True
* step text: `* **Yesterday's buyers:** The day before had 50 buyers, and yesterday had half that number, so yesterday had 50 / 2 = 25 buyers.`
* parsed: op=/ a=50 b=2 result=25 a_ref=('slot', 1) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 3) (*, ('const', Fraction(1, 2)), ('slot', 1), 25)
* error type: planning  extra_valid=False
* gold program: 0: 1/2 * 50 = 25; 1: 40 + 25 = 65; 2: 65 + 25 = 90; 3: 90 + 50 = 140
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5510 terse_p0 trace_correct=True
* step text: `* Ryan has 25 + 20 = 45 stickers.`
* parsed: op=+ a=25 b=20 result=45 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (+, ('slot', 0), ('slot', 1), 45)
* error type: none  extra_valid=False
* gold program: 0: 25 + 20 = 45; 1: 45 - 10 = 35; 2: 25 + 45 = 70; 3: 70 + 35 = 105
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5956 terse_p0 trace_correct=True
* step text: `* Cost of gasoline: 20 liters * $0.90/liter = $18.00`
* parsed: op=* a=20 b=9/10 result=18 a_ref=('const', Fraction(20, 1)) b_ref=('slot', 4) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (*, ('slot', 4), ('res', 1), 18)
* error type: none  extra_valid=False
* gold program: 0: 150 * 2 = 300; 1: 300 / 15 = 20; 2: 9/10 * 20 = 18; 3: 50 + 18 = 68; 4: 90 - 68 = 22
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-12 terse_p0 trace_correct=True
* step text: `* Randy has 5 less than half the number of mango trees, so he has 30 - 5 = 25 coconut trees.`
* parsed: op=- a=30 b=5 result=25 a_ref=('res', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('res', 0), ('slot', 1), 25)
* error type: none  extra_valid=False
* gold program: 0: 60 / 2 = 30; 1: 30 - 5 = 25; 2: 60 + 25 = 85
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6191 terse_p0 trace_correct=True
* step text: `* **Calculate the number of sold seats:** 60,000 seats * 75% = 45,000 seats`
* parsed: op=* a=60000 b=3/4 result=45000 a_ref=('slot', 0) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 1), ('slot', 0), 45000)
* error type: none  extra_valid=False
* gold program: 0: 3/4 * 60000 = 45000; 1: 45000 - 5000 = 40000
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3691 marked_p0 trace_correct=True
* step text: `Step 6: Calculate the earnings from tomatoes: 16 baskets * $6/basket = $96`
* parsed: op=* a=16 b=6 result=96 a_ref=('slot', 3) b_ref=('slot', 6) compound=False ambiguous=False
* aligned gold step: 5 (tier 1) (*, ('res', 3), ('slot', 6), 96)
* error type: none  extra_valid=False
* gold program: 0: 5 * 14 = 70; 1: 7 * 16 = 112; 2: 70 / 7 = 10; 3: 112 / 7 = 16; 4: 10 * 9 = 90; 5: 16 * 6 = 96; 6: 90 + 96 = 186
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3155 terse_p0 trace_correct=False
* step text: `* Thomas and his friends take 1/4 of 200 bars: (1/4) * 200 = 50 bars`
* parsed: op=* a=1/4 b=200 result=50 a_ref=('slot', 2) b_ref=('slot', 0) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (*, ('slot', 2), ('slot', 0), 50)
* error type: none  extra_valid=False
* gold program: 0: 1/4 * 200 = 50; 1: 200 - 50 = 150; 2: 50 / 5 = 10; 3: 150 + 5 = 155; 4: 50 - 5 = 45; 5: 155 - 45 = 110
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-1265 terse_p0 trace_correct=True
* step text: `* Total cost: $24 * 2 = $48`
* parsed: op=* a=24 b=2 result=48 a_ref=('slot', 2) b_ref=('slot', 3) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('slot', 2), ('slot', 3), 48)
* error type: none  extra_valid=False
* gold program: 0: 30 * 4 = 120; 1: 24 * 2 = 48; 2: 120 + 48 = 168; 3: 168 / 6 = 28
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4459 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the value of the dimes. 12 dimes * $0.10/dime = $1.20`
* parsed: op=* a=12 b=1/10 result=6/5 a_ref=('slot', 2) b_ref=('const', Fraction(1, 10)) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (*, ('slot', 2), ('const', Fraction(1, 10)), 6/5)
* error type: none  extra_valid=False
* gold program: 0: 10 * 1/4 = 5/2; 1: 12 * 1/10 = 6/5; 2: 4 - 5/2 = 3/2; 3: 3/2 - 6/5 = 3/10; 4: 3/10 / 1/20 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-395 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the total number of elephants remaining after the exodus.
= 30,000 elephants - 11,520 elephants = 18,480 elephants`
* parsed: op=- a=30000 b=11520 result=18480 a_ref=('slot', 0) b_ref=('const', Fraction(11520, 1)) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('slot', 0), ('res', 0), 18480)
* error type: none  extra_valid=False
* gold program: 0: 4 * 2880 = 11520; 1: 30000 - 11520 = 18480; 2: 28980 - 18480 = 10500; 3: 10500 / 7 = 1500
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-333 marked_p0 trace_correct=False
* step text: `Step 2: Calculate the total number of chocolates Kantana bought last Saturday: 3 + 10 = 13 chocolates`
* parsed: op=+ a=3 b=10 result=13 a_ref=('res', 0) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 3 (tier 4) (+, ('slot', 2), ('res', 2), 14)
* error type: binding  extra_valid=False
* gold program: 0: 1 * 4 = 4; 1: 2 * 4 = 8; 2: 1 * 4 = 4; 3: 10 + 4 = 14; 4: 14 + 8 = 22
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-346 marked_p0 trace_correct=True
* step text: `Step 4: Calculate the total area of the walls.
Total area = 6 square meters + 6 square meters + 10 square meters + 8 square meters = 30 square meters`
* parsed: op=None a=None b=None result=30 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 5 (tier 3) (+, ('res', 4), ('res', 3), 30)
* error type: None  extra_valid=False
* gold program: 0: 3 * 2 = 6; 1: 6 * 2 = 12; 2: 5 * 2 = 10; 3: 4 * 2 = 8; 4: 12 + 10 = 22; 5: 22 + 8 = 30; 6: 30 / 2 = 15
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-117 terse_p0 trace_correct=False
* step text: `* **Total blowfish:** Since the aquarium holds an equal number of clownfish and blowfish, and there are 100 fish total, there must be 100 / 2 = 50 blowfish.`
* parsed: op=/ a=100 b=2 result=50 a_ref=('slot', 1) b_ref=('const', Fraction(2, 1)) compound=False ambiguous=False
* aligned gold step: 0 (tier 1) (/, ('slot', 1), ('const', Fraction(2, 1)), 50)
* error type: none  extra_valid=False
* gold program: 0: 100 / 2 = 50; 1: 50 - 26 = 24; 2: 24 / 3 = 8; 3: 24 - 8 = 16
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5368 marked_p0 trace_correct=True
* step text: `Step 4: Calculate the number of meat pieces per slice: Lisa has 132 pieces of meat and 6 slices of pizza, so each slice has 132 / 6 = 22 pieces of meat.`
* parsed: op=/ a=132 b=6 result=22 a_ref=('rres', 2) b_ref=('slot', 2) compound=False ambiguous=False
* aligned gold step: 6 (tier 3) (+, ('res', 5), ('res', 4), 22)
* error type: planning  extra_valid=False
* gold program: 0: 30 * 2 = 60; 1: 30 + 12 = 42; 2: 30 / 6 = 5; 3: 60 / 6 = 10; 4: 42 / 6 = 7; 5: 5 + 10 = 15; 6: 15 + 7 = 22
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-2933 terse_p0 trace_correct=True
* step text: `* **Subtract the books taken out on Friday:** 64 - 35 = 29`
* parsed: op=- a=64 b=35 result=29 a_ref=('res', 1) b_ref=('slot', 3) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (-, ('res', 1), ('slot', 3), 29)
* error type: none  extra_valid=False
* gold program: 0: 235 - 227 = 8; 1: 8 + 56 = 64; 2: 64 - 35 = 29
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5808 terse_p0 trace_correct=True
* step text: `*  Fiona's time is 8 hours - 2 hours = 6 hours.`
* parsed: op=- a=8 b=2 result=6 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('slot', 0), ('res', 0), 6)
* error type: none  extra_valid=False
* gold program: 0: 8 / 4 = 2; 1: 8 - 2 = 6; 2: 6 * 60 = 360
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-374 terse_p0 trace_correct=True
* step text: `* **Total buyers:**  The total number of buyers over the three days is 50 + 25 + 65 = 140 buyers.`
* parsed: op=None a=None b=None result=140 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 3 (tier 3) (+, ('res', 2), ('slot', 1), 140)
* error type: None  extra_valid=False
* gold program: 0: 1/2 * 50 = 25; 1: 40 + 25 = 65; 2: 65 + 25 = 90; 3: 90 + 50 = 140
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-7452 terse_p0 trace_correct=True
* step text: `* Total value = $600 + $600 + $150 = $1350`
* parsed: op=None a=None b=None result=1350 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: 5 (tier 3) (+, ('res', 4), ('res', 3), 1350)
* error type: None  extra_valid=False
* gold program: 0: 900 / 3 = 300; 1: 300 * 2 = 600; 2: 300 * 2 = 600; 3: 300 / 2 = 150; 4: 600 + 600 = 1200; 5: 1200 + 150 = 1350
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3032 terse_p0 trace_correct=True
* step text: `* **Total harvest on Wednesday and Thursday:**  He harvested a total of 400 kg + 200 kg = 600 kg on Wednesday and Thursday.`
* parsed: op=+ a=400 b=200 result=600 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 5) (-, ('slot', 1), ('slot', 0), 1600)
* error type: planning  extra_valid=False
* gold program: 0: 1/2 * 400 = 200; 1: 2000 - 400 = 1600; 2: 1600 - 200 = 1400; 3: 1400 - 700 = 700
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5992 terse_p0 trace_correct=False
* step text: `* **Calculate the weight of the leaves per day:** 1000 leaves * 1 pound/1000 leaves = 1 pound`
* parsed: op=None a=None b=None result=1 a_ref=None b_ref=None compound=True ambiguous=False
* aligned gold step: None (tier None) 
* error type: None  extra_valid=False
* gold program: 0: 500 * 1000 = 500000; 1: 500000 / 100 = 5000
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6967 terse_p0 trace_correct=True
* step text: `* Together, they have 44 + 20 = 64 pencils.`
* parsed: op=+ a=44 b=20 result=64 a_ref=('res', 1) b_ref=('slot', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (+, ('res', 1), ('slot', 1), 64)
* error type: none  extra_valid=False
* gold program: 0: 2 * 20 = 40; 1: 40 + 4 = 44; 2: 44 + 20 = 64
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3234 marked_p0 trace_correct=True
* step text: `Step 2: Calculate the amount left for the other grandchildren.
The total amount left for the other grandchildren is $124,600 - $62,300 = $62,300.`
* parsed: op=- a=124600 b=62300 result=62300 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 5) (/, ('res', 0), ('slot', 1), 6230)
* error type: planning  extra_valid=False
* gold program: 0: 1/2 * 124600 = 62300; 1: 62300 / 10 = 6230
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4870 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the revenue from children's tickets.
The number of children's tickets sold is 28.
The revenue from children's tickets is 28 * $13 = $364.`
* parsed: op=* a=28 b=13 result=364 a_ref=('slot', 5) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (*, ('slot', 5), ('res', 1), 364)
* error type: none  extra_valid=False
* gold program: 0: 183 * 26 = 4758; 1: 26 / 2 = 13; 2: 28 * 13 = 364; 3: 4758 + 364 = 5122
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3183 marked_p0 trace_correct=True
* step text: `Step 2: Find the number of students taking music.
400 - 320 = 80`
* parsed: op=- a=400 b=320 result=80 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 3) (-, ('res', 0), ('slot', 2), 80)
* error type: binding  extra_valid=False
* gold program: 0: 400 - 120 = 280; 1: 280 - 200 = 80; 2: 1/5 * 100 = 20
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-3005 marked_p0 trace_correct=True
* step text: `Step 3: Calculate the total amount spent.
Together they spent $400 + $560 = $960.`
* parsed: op=+ a=400 b=560 result=960 a_ref=('slot', 1) b_ref=('res', 1) compound=False ambiguous=False
* aligned gold step: 2 (tier 1) (+, ('slot', 1), ('res', 1), 960)
* error type: none  extra_valid=False
* gold program: 0: 2/5 * 400 = 160; 1: 400 + 160 = 560; 2: 400 + 560 = 960
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-6468 marked_p0 trace_correct=True
* step text: `Step 2: Add the price increase to the original price.
$2000 + $40 = $2040`
* parsed: op=+ a=2000 b=40 result=2040 a_ref=('slot', 0) b_ref=('res', 0) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (+, ('slot', 0), ('res', 0), 2040)
* error type: none  extra_valid=False
* gold program: 0: 1/50 * 2000 = 40; 1: 2000 + 40 = 2040
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-4051 terse_p0 trace_correct=True
* step text: `* $55 / 5 people = $11/person`
* parsed: op=/ a=55 b=5 result=11 a_ref=('const', Fraction(55, 1)) b_ref=('slot', 1) compound=False ambiguous=True
* aligned gold step: 6 (tier 1) (/, ('res', 5), ('slot', 1), 11)
* error type: none  extra_valid=False
* gold program: 0: 5 * 5 = 25; 1: 5/2 * 4 = 10; 2: 2 * 5 = 10; 3: 10 + 25 = 35; 4: 35 + 10 = 45; 5: 45 + 10 = 55; 6: 55 / 5 = 11
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-331 terse_p0 trace_correct=False
* step text: `* One mile less than double: 8 miles - 1 mile = 7 miles`
* parsed: op=- a=8 b=1 result=7 a_ref=('res', 0) b_ref=('const', Fraction(1, 1)) compound=False ambiguous=False
* aligned gold step: 1 (tier 1) (-, ('res', 0), ('const', Fraction(1, 1)), 7)
* error type: none  extra_valid=False
* gold program: 0: 4 * 2 = 8; 1: 8 - 1 = 7; 2: 16 - 7 = 9; 3: 9 - 4 = 5; 4: 3 * 5 = 15; 5: 15 * 2 = 30; 6: 30 / 5 = 6
* manual check: [ ] correct  [ ] wrong (note):

## gsm8k-train-5746 marked_p0 trace_correct=False
* step text: `Step 5: Calculate the total cost of tickets.
$12 + $48 = $60`
* parsed: op=+ a=12 b=48 result=60 a_ref=('res', 0) b_ref=('res', 1) compound=False ambiguous=True
* aligned gold step: None (tier None) 
* error type: planning  extra_valid=False
* gold program: 0: 12 * 1 = 12; 1: 4 * 3 = 12; 2: 12 + 12 = 24
* manual check: [ ] correct  [ ] wrong (note):
