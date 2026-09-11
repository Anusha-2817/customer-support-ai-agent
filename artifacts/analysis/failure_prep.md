# Failure-analysis preparation: system A, 130 uniform (headline) cases

Evidence for choosing the top five failure modes. Golden labels are used for analysis only.

## 1. Who escalates

- The model itself asked for a human on **16/130** cases; gold labels escalate **65/130**.
- Escalation sources: (auto): 97; guard: 17; model: 14; invalid_output: 2

## 2. Missed escalations (gold escalate, A auto-sent)

**38** of 65 must-escalate cases were auto-sent. By gold reason: `disputed_policy_claim` 14, `reputational_risk` 8, `repeated_contact` 5, `needs_booking_access` 5, `compensation_dispute` 4, `safety_medical` 2

### `disputed_policy_claim`
- **78379**: "shocking service. How can CANCELLING one leg of my journey result in a £70 fee?! I refuse to pay this and need to keep my return journey. Sort this out BA, I chose to fly because you are the premium …"
  - gold: intent `booking_change`, escalate `disputed_policy_claim` | A: intent `service_complaint`, auto (by none)
  - A's draft: "Hi, I understand this is frustrating. As a premium airline, we strive for excellence. Please DM me your booking details so we can look into this further."
- **138758**: "£360 to change a £140 ticket to a different flight same day for “one of your most valued Gold Card customers”, who’s flown 150 times this year with you already - customer loyalty... you’re doing it w…"
  - gold: intent `booking_change`, escalate `disputed_policy_claim` | A: intent `booking_change`, auto (by none)
  - A's draft: "I understand this is frustrating, but the change fee is based on the rules of the ticket purchased. For Gold Card members, we aim to be fair and consistent. Please DM me for further assistance."
- **2724371** (thread): "But if that’s done then the bags presumably can’t get to INV without them?"
  - gold: intent `baggage`, escalate `disputed_policy_claim` | A: intent `flight_disruption`, auto (by none)
  - A's draft: "If someone in your group decides to leave INV, their bags will be checked through to LHR. If you need to change your booking to end in LHR, our staff can assist with that. Please let me know if you need any further help."
- **455270** (thread): "You've not answered all my questions - I take my responsibility for being 5min late, but your staff could have easily (assessed it's not busy and) check me in."
  - gold: intent `travel_info`, escalate `disputed_policy_claim` | A: intent `service_complaint`, auto (by none)
  - A's draft: "I understand you feel let down by the situation. Could you please provide your full name and booking reference so we can look into this further?"

### `reputational_risk`
- **2744927**: "In the 60s, we made our second-class citizens sit in the back of the bus. Today we make them sit in the back of the plane. #CapitalistPatriarchy is modern day Jim Crow. @delta @AmericanAir <link>"
  - gold: intent `service_complaint`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none)
  - A's draft: "We understand this can be frustrating. Can you please share which flight this was so we can look into it further?"
- **705392**: "Dear - decided to give you one last try after years of frustration. You have just cancelled my family's flights for next summer and feel that a flight 10hrs later from a different airport is ok to of…"
  - gold: intent `flight_disruption`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none)
  - A's draft: "I understand this is frustrating. Can you please share what happened during your flight and how we can improve for next time?"
- **578019** (thread): "Will do - just read my later flight is delayed now too. I understand your system has to have a cut off - but when two passengers are travelling together some manual intervention in the system should …"
  - gold: intent `flight_disruption`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none)
  - A's draft: "Hi, I understand your system has to have a cut-off, but in this case, it caused a significant inconvenience. When two passengers are traveling together, shouldn't there be some manual intervention to allow for a connection? This delay was …"
- **532326** (thread): "No, nothing. Could you check and let me know how to avoid another bad flight with BA?"
  - gold: intent `booking_change`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none)
  - A's draft: "Hi Luca, I understand you're frustrated. Our Customer Experience team will look into how to avoid such issues on your return flight. Please DM me your booking reference so we can assist further."

### `repeated_contact`
- **643111**: "how comes I returned from a holiday in September that I made a complaint about and am yet to receive a resolution other than for the flight that the staff member was rude but you didn’t seem to care"
  - gold: intent `refund_claim_followup`, escalate `repeated_contact` | A: intent `service_complaint`, auto (by none)
  - A's draft: "Hi, I've been waiting for a resolution to my complaint from September. It's been a month and I'm still not happy. Can you please provide a response or direct me to the right team to follow up? Thanks."
- **2880668** (thread): "That's what I did but It doesn't work. I even uploaded the receipts but It doesn't work It sends me an error message. Could you send me the proper link here ? Regards"
  - gold: intent `baggage`, escalate `repeated_contact` | A: intent `refund_claim_followup`, auto (by none)
  - A's draft: "Hi, I understand you're having trouble submitting receipts. Please send a screenshot of the error message and your details so we can look into this further. We'll get you the right link or process. Thanks."
- **2844219**: "no one contacted me and no one is answering the phone number! What can i do? <link>"
  - gold: intent `contact_dm`, escalate `repeated_contact` | A: intent `contact_dm`, auto (by none)
  - A's draft: "Hi, I've tried all the numbers provided but can't get through. Could you please DM me your full name and email so we can check this for you? #BritishAirways"
- **542816** (thread): "The guys who are here taking the strain are but they should have adequate resources. If this was a one off that would be fine but it is every time we land in terminal 5."
  - gold: intent `service_complaint`, escalate `repeated_contact` | A: intent `service_complaint`, auto (by none)
  - A's draft: "I understand this is frustrating. We're working on improving the resources at Terminal 5. Please share your feedback with our Customer Experience team so we can address this internally. Thank you for bringing this to our attention."

## 3. Guard false alarms (gold auto, escalated by the guard)

The guard fired on 22 headline cases; 17 of them are gold escalations (precision 0.77). **4** false alarms, by first rule: `repeat_contact_words` 2, `money_claim_words` 1, `high_visibility_words` 1

- **1844342** (thread): "Can you give me an ETA on “soon”?"
  - gold: intent `refund_claim_followup`, auto | A: intent `refund_claim_followup`, escalate `compensation_dispute` (by guard) | guard evidence: money_claim_words: 'still waiting for a refund'; repeat_contact_words: 'still waiting'
  - A's draft: "Hi, we understand your concern. A member of our team will be in touch as soon as we have more information. Please keep an eye on your inbox."
- **2842272** (thread): "Big thumbs up to BA’s social media team. Sorted out seats on up coming flights to and from South America, muchas gracias amigos :)⭐️"
  - gold: intent `other_non_actionable`, auto | A: intent `travel_info`, escalate `reputational_risk` (by guard) | guard evidence: high_visibility_words: 'media'
  - A's draft: "Thanks for the kind words, Sarah! We're glad to hear your flight was fabulous. If you need any assistance with your next booking, feel free to DM us."
- **2849672**: "Hi there, I have a question regarding baggage policy that is still not clear to my. Maybe you can help me out :) thanks"
  - gold: intent `baggage`, auto | A: intent `travel_info`, escalate `repeated_contact` (by guard) | guard evidence: repeat_contact_words: 'still not'
  - A's draft: "Hi there, I'd be happy to help with your baggage policy query. Do you have your booking reference and full name handy? If not, we can DM you the details."
- **2830975**: "Getting ready for a crazy five weeks of travel. Flying almost 36K miles & nearly 72 hours on starting on Wednesday. It's a good thing I like flying... 😱"
  - gold: intent `other_non_actionable`, auto | A: intent `travel_info`, escalate `repeated_contact` (by guard) | guard evidence: repeat_contact_words: 'five weeks'
  - A's draft: "That's a busy travel schedule ahead! Where are you flying to and which flights are you looking forward to? Looking forward to hearing more about your adventures!"

## 4. Intent confusions

A is wrong on **78/130**; B1 (TF-IDF) is right on **41** of those.
Top confusions (gold -> A): `other_non_actionable` -> `travel_info` 13, `flight_disruption` -> `service_complaint` 10, `other_non_actionable` -> `service_complaint` 8, `refund_claim_followup` -> `service_complaint` 7, `booking_change` -> `service_complaint` 5, `travel_info` -> `service_complaint` 4

### gold `other_non_actionable` -> A `travel_info`
- **2842272** (thread): "Big thumbs up to BA’s social media team. Sorted out seats on up coming flights to and from South America, muchas gracias amigos :)⭐️"
  - gold: intent `other_non_actionable`, auto | A: intent `travel_info`, escalate `reputational_risk` (by guard) | B1: `booking_change`
  - A's draft: "Thanks for the kind words, Sarah! We're glad to hear your flight was fabulous. If you need any assistance with your next booking, feel free to DM us."
- **2731023**: "flew last week on BA057 world traveller + cabin. Fantastic crew- helpful, always smiling- a v pleasant experience"
  - gold: intent `other_non_actionable`, auto | A: intent `travel_info`, auto (by none) | B1: `service_complaint`
  - A's draft: "Hi there, your flight was a fantastic experience with a wonderful crew. We're pleased to hear that. Looking forward to welcoming you on board again soon!"
- **700115**: "And relax! (@ BA Galleries Club Lounge South - in Hounslow, London) <link> <link>"
  - gold: intent `other_non_actionable`, auto | A: intent `travel_info`, auto (by none) | B1: `travel_info`
  - A's draft: "Thanks for sharing, Sarah. I hope you had a relaxing time in the lounge and enjoyed your flight."

### gold `flight_disruption` -> A `service_complaint`
- **705392**: "Dear - decided to give you one last try after years of frustration. You have just cancelled my family's flights for next summer and feel that a flight 10hrs later from a different airport is ok to of…"
  - gold: intent `flight_disruption`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none) | B1: `service_complaint`
  - A's draft: "I understand this is frustrating. Can you please share what happened during your flight and how we can improve for next time?"
- **2877950** (thread): "Staff on the plane were great but the follow-up about getting booked onto next flight has been a disaster..no plan..told to wait in hotel..very poor customer service"
  - gold: intent `flight_disruption`, escalate `needs_booking_access` | A: intent `service_complaint`, auto (by none) | B1: `service_complaint`
  - A's draft: "I understand this has been frustrating. Have you had any updates on your next flight? If you need further assistance, please DM me."
- **87345**: "Hi I'm on BA107 from Heathrow. I'm finding it distressing that the plane has a damaged door yet you have been deciding for an hour and a half whether it is safe to take off or not. Does not make me f…"
  - gold: intent `flight_disruption`, escalate `safety_medical` | A: intent `service_complaint`, auto (by none) | B1: `flight_disruption`
  - A's draft: "I understand this is distressing. Our team is working hard to ensure your safety and comfort. Please stay in the cabin and follow the crew's instructions. We will keep you updated."

### gold `other_non_actionable` -> A `service_complaint`
- **24012** (thread): "I am never flying with you again, and will make sure none of my family or my workplace do either, absolutely pathetic"
  - gold: intent `other_non_actionable`, escalate `reputational_risk` | A: intent `service_complaint`, escalate `compensation_dispute` (by guard) | B1: `service_complaint`
  - A's draft: "I understand this has been a very frustrating experience. Please know we value your feedback and will look into improving our processes. Is there anything specific you would like us to do to make amends, such as a voucher for future travel…"
- **160224** (thread): "Just returned in what most call premium economy and it was superior to business class as had a great crew, stayed away from beef and scrambled eggs......chicken and a toasted cheese/ham toastie is a …"
  - gold: intent `other_non_actionable`, auto | A: intent `service_complaint`, auto (by none) | B1: `service_complaint`
  - A's draft: "Hi Sharon, I understand you had a different experience in Business Class. We appreciate your feedback and will pass it on to the relevant team. If you have any further concerns, feel free to DM us."
- **478022**: "Shout out to Chris Moyes who was on the Mumbai Heathrow flight today! Impeecible service, great attitude and very considerate! Is there any way I can send appreciation to him that would add to his in…"
  - gold: intent `other_non_actionable`, auto | A: intent `service_complaint`, auto (by none) | B1: `travel_info`
  - A's draft: "Thanks for your kind words, Chris! We're glad you had such a great experience. If you'd like to send a personal message to the crew, you can do so through our internal channels."

Gold intent distribution vs A's predictions: `other_non_actionable` gold 24 / A 1, `flight_disruption` gold 18 / A 12, `refund_claim_followup` gold 17 / A 13, `travel_info` gold 16 / A 24, `booking_change` gold 14 / A 15, `baggage` gold 13 / A 3, `service_complaint` gold 12 / A 46, `website_app` gold 9 / A 6, `contact_dm` gold 4 / A 7, `loyalty_avios` gold 3 / A 1

## 5. Reply-check failures among A's auto-sent replies

21 of 97 auto-sent drafts fail at least one check: `public_pii_request` 10, `fabricated` 6, `too_long` 3, `commitments` 2

### `public_pii_request`
- **455270** (thread): "You've not answered all my questions - I take my responsibility for being 5min late, but your staff could have easily (assessed it's not busy and) check me in."
  - gold: intent `travel_info`, escalate `disputed_policy_claim` | A: intent `service_complaint`, auto (by none)
  - A's draft: "I understand you feel let down by the situation. Could you please provide your full name and booking reference so we can look into this further?"
- **2771528**: "can you advise on your website where I need to claim for a 5 hour delay to BA0217 on 17th November. On your website on the claim for delays it states for food, accommodation etc and receipts. I want …"
  - gold: intent `refund_claim_followup`, auto | A: intent `refund_claim_followup`, auto (by none)
  - A's draft: "Hi, for a 5-hour delay, you can claim compensation. Please provide your full name, booking reference and email address for us to process your claim. Thanks."
- **99134**: "I had to cancel my family's flight to Inverness and Edinburg this July. I contacted British airways and the gentleman on the phone that day assured me the flight would be refunded. I cannot seem to f…"
  - gold: intent `refund_claim_followup`, escalate `disputed_policy_claim` | A: intent `refund_claim_followup`, auto (by none)
  - A's draft: "Hi, I understand you're looking for proof of your refund. Please provide your booking reference number and any other details you have, and we'll check it out for you."

### `fabricated`
- **2743945**: "Hi, I'm flying BA8491 from LCY on Saturday. Can I check my bicycle (in a transport box) and where and how? #neverdonethisbefore"
  - gold: intent `baggage`, auto | A: intent `baggage`, auto (by none) | flagged: ['£15.']
  - A's draft: "Hi, for your bicycle, you can check it in as a personal item. Please ensure it's in a transport box. You'll need to pay for the additional piece. The cost for an extra piece is £15. If you need further assistance, please DM me your full na…"
- **1647095**: "100 Canadian dollars for 3kg overweight bag.... really???. #ba84"
  - gold: intent `baggage`, escalate `disputed_policy_claim` | A: intent `baggage`, auto (by none) | flagged: ['£65']
  - A's draft: "Hi there, I understand this is disappointing. Our standard overweight baggage charge is £65 per bag. If you have any further questions or need assistance, please DM us for more details."
- **122496**: "Are you guys having a black Friday sale... I need a holiday asap"
  - gold: intent `travel_info`, auto | A: intent `travel_info`, auto (by none) | flagged: ['<link>']
  - A's draft: "Hi there. We don't have a date yet for Black Friday sales, but you can register on <link> for all our special offers. Many thanks."

### `too_long`
- **578019** (thread): "Will do - just read my later flight is delayed now too. I understand your system has to have a cut off - but when two passengers are travelling together some manual intervention in the system should …"
  - gold: intent `flight_disruption`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none)
  - A's draft: "Hi, I understand your system has to have a cut-off, but in this case, it caused a significant inconvenience. When two passengers are traveling together, shouldn't there be some manual intervention to allow for a connection? This delay was …"
- **475017**: "I can't login with the BA Android app for several days with invalid grant. See screenshot <link>"
  - gold: intent `website_app`, auto | A: intent `website_app`, auto (by none)
  - A's draft: "Hi [Customer Name], we're sorry to hear you're having trouble with the app. Have you tried updating to the latest version and then deleting and reinstalling it? If you're still having issues, we recommend using Manage My Booking on our des…"
- **2888639**: "Is there any chance you could give me some advice ? My son is interested in becoming a pilot, he leaves school 2018 & is expected A's & A*'s what should his next steps be to reach his dream?"
  - gold: intent `travel_info`, auto | A: intent `travel_info`, auto (by none)
  - A's draft: "Hi, I understand you're looking for advice on your son's future as a pilot. British Airways has a dedicated Aviation Academy where children can become pilots or part of the cabin crew. They offer a great introduction to the industry. I hop…"

### `commitments`
- **2849681** (thread): "November 21st - one week ago."
  - gold: intent `refund_claim_followup`, auto | A: intent `refund_claim_followup`, auto (by none) | flagged: ['will be in touch']
  - A's draft: "Hi [Customer], thank you for reaching out. Your claim case number is [Customer's case number]. Our Customer Relations team will be in touch to update you. If you have any other questions, feel free to DM me."
- **2766080**: "Will never travel BA again after response from their customer services. you win- I give up . Still out of pocket."
  - gold: intent `refund_claim_followup`, escalate `reputational_risk` | A: intent `service_complaint`, auto (by none) | flagged: ['will be in touch']
  - A's draft: "I'm really sorry to hear that you're having such a difficult time with our service. Our Customer Relations team will be in touch to help resolve this for you."

## 6. Invalid model outputs

2 of 130:
- 87329: ["invalid intent 'bag'; escalated"]
- 482195: ["invalid intent 'bag'; escalated"]

## 7. Is the model's own confidence informative?

intent_confidence values: 1.0: 128, 0.95: 2
Accuracy when confidence = 1.0: 50/128

