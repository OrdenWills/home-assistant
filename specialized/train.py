"""
generate_stage2_dataset.py  —  Smart Home SFT  |  Stage 2  v12

═══════════════════════════════════════════════════════════════════════
v11 → v12 CHANGES  (production hardening — gap closure from 42-log audit)
═══════════════════════════════════════════════════════════════════════

FIX-A  "Back" synonym collision.
       v11's gen_back_synonym_disambiguation contained a negative think
       trace that explicitly mentioned the speaker tool ("It does NOT mean
       speaker 'previous'").  Negative examples in think traces cause the
       model to associate the wrong action with the phrase.  Replaced with
       a purely positive trace that reasons from context + log evidence.

FIX-B  Positive-only think traces throughout.
       Scanned every generator for negative tool mentions (e.g.
       "'stop', NOT action='next'").  Replaced all such strings with
       affirmative reasoning.  The model now learns WHAT to do, not
       what NOT to do.

FIX-C  gen_rejections extended to non-light devices.
       Previously only trained unsupported_device for lights.  The model
       was hallucinating doors/fans/TVs in rooms that don't have them.
       Added door, TV, and fan branches with the same rejection pattern.

FIX-D  Auto ACTION REQUIRED / ACTION NOT REQUIRED marker in build_ex.
       All think traces now end with a rigid syntactic trigger that the
       model can use to decide whether to open a tool_call block.  This
       is a single change in build_ex — every generator inherits it.
       This closes the "empty tool_calls" and "JSON bleed" failure modes
       observed in turns 11-14, 56, and 65 of the audit logs.

FIX-E  CATEGORY_PLAN updated.
       gen_back_synonym_disambiguation and gen_relative_state_clauses
       (introduced in v11) are now correctly wired into the plan with
       explicit targets (800 and 1 000 respectively).

ALL v11 hybrid exhaustive generators (400 + 240 + 400 + 320) preserved.
ALL v9/v10 gap fixes (A–M) preserved.
ALL v10 transaction log bracket notation preserved.
"""

import json, random, math, re, itertools
from collections import Counter, defaultdict

random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# UNIVERSE
# ══════════════════════════════════════════════════════════════════════

ALL_ROOMS     = ["living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"]
ALL_DOORS     = ["front", "back", "garage", "side", "bedroom", "bathroom",
                 "office", "kitchen", "living_room"]
SCENES        = ["movie_night", "bedtime", "morning", "away", "party"]
TV_ROOMS      = ["living_room", "bedroom", "office"]
FAN_ROOMS     = ["living_room", "bedroom", "kitchen", "office"]
SPEAKER_ROOMS = ["living_room", "bedroom", "kitchen", "office", "hallway"]
MIN_T, MAX_T  = 60, 80

ROOM_ALIASES: dict[str, list[str]] = {
    "living_room": ["living room", "lounge", "front room", "sitting room",
                    "living area", "family room", "main room"],
    "bedroom":     ["bedroom", "master bedroom", "my room", "sleeping area"],
    "kitchen":     ["kitchen", "kitchenette", "cooking area"],
    "bathroom":    ["bathroom", "bath", "washroom", "restroom", "lavatory", "toilet"],
    "office":      ["office", "home office", "study", "workspace", "den"],
    "hallway":     ["hallway", "hall", "corridor", "entry", "entryway", "foyer"],
}
DOOR_ALIASES: dict[str, list[str]] = {
    "front":       ["front door", "main door", "front entrance", "front"],
    "back":        ["back door", "rear door", "back entrance"],
    "garage":      ["garage door", "garage"],
    "side":        ["side door", "side entrance"],
    "bedroom":     ["bedroom door"],
    "bathroom":    ["bathroom door", "bath door"],
    "office":      ["office door"],
    "kitchen":     ["kitchen door"],
    "living_room": ["living room door", "lounge door"],
}
DOOR_DISPLAY = {d: DOOR_ALIASES[d][0] for d in ALL_DOORS}
ROOM_DISPLAY = {r: ROOM_ALIASES[r][0] for r in ALL_ROOMS}

SCENE_RESP = {
    "movie_night": "Movie Night scene activated.",
    "bedtime":     "Bedtime scene activated.",
    "morning":     "Morning scene activated.",
    "away":        "Away mode activated.",
    "party":       "Party scene activated.",
}

SCENE_TRIGGERS = {
    "movie_night": [
        "movie night", "movie mode", "let's watch a movie", "cinema mode", 
        "time for a movie", "movie time", "ready for a movie", "start movie night",
        "turn on the cinema lights", "set up the theater", "popcorn time", 
        "let's binge watch something", "time to relax with a film", 
        "switch to theater mode", "movie marathon time"
    ],
    "bedtime": [
        "bedtime", "going to bed", "sleep mode", "night mode", "time for bed", 
        "I want to sleep", "i am going to sleep", "ready for bed", 
        "turn in for the night", "shutting down for the night", "lights out", 
        "goodnight everyone", "time to hit the hay", "activating sleep settings", 
        "drifting off to sleep"
    ],
    "morning": [
        "good morning", "morning mode", "rise and shine", "wake up mode", 
        "it's morning already", "i just woke up", "waking up", "time to wake up", 
        "start my day", "good morning sunshine", "let's start the day", 
        "morning routine", "i am awake now", "time to get moving", 
        "greet the day"
    ],
    "away": [
        "away mode", "I'm leaving", "heading out", "going out", "leaving home", 
        "am going out", "i am leaving now", "see you later", "bye bye", 
        "goodbye house", "securing the home", "vacation mode", "i am off", 
        "nobody is home", "all clear for departure"
    ],
    "party": [
        "party mode", "having guests", "let's party", "party time", 
        "guests are here", "time to celebrate", "let's get this party started", 
        "turn up the music", "entertaining guests", "set the party mood", 
        "having friends over", "let's have a blast", "it's time to entertain", 
        "start the celebration", "party lights on"
    ],
}

OFF_TOPIC_PHRASES = [
    "What's the weather like?", "Order me a pizza.", "What time is it?",
    "Set an alarm for 7 AM.", "Tell me a joke.", "What's today's news?",
    "How do I cook pasta?", "Search for flights.", "Call Dad.",
    "Translate hello to Spanish.", "What's my schedule today?"
]
UNSUPPORTED_FEATURE_PHRASES = [
    "Dim the {r} lights to 50%.", "Set the {r} lights to 30%.",
    "Change the {r} light color to blue.", "Make the {r} lights softer.",
    "Half brightness in the {r}.", "Turn the {r} lights to red."
]
UNSUPPORTED_APPLIANCES = [
    ("coffee maker",    ["Start the coffee maker.", "Turn on the coffee maker."]),
    ("microwave",       ["Turn on the microwave.", "Start the microwave."]),
    ("oven",            ["Preheat the oven to 350.", "Turn on the oven."]),
    ("dishwasher",      ["Start the dishwasher.", "Run the dishwasher."]),
    ("washing machine", ["Start the washing machine.", "Run the washing machine."]),
    ("dryer",           ["Start the dryer.", "Turn on the dryer."]),
    ("robot vacuum",    ["Start the robot vacuum.", "Run the robot vacuum."]),
    ("air purifier",    ["Turn on the air purifier.", "Start the air purifier."]),
    ("humidifier",      ["Turn on the humidifier.", "Start the humidifier."]),
    ("sprinklers",      ["Turn on the sprinklers.", "Activate the sprinklers."]),
    ("projector",       ["Turn on the projector.", "Start the projector."]),
    ("fireplace",       ["Turn on the fireplace.", "Start the fireplace."]),
    ("blinds",          ["Close the blinds.", "Open the blinds."]),
    ("curtains",        ["Open the curtains.", "Close the curtains."]),
    ("grill",           ["Fire up the grill.", "Turn on the grill."]),
]


LOCAL_MUSIC = [
        'Truth In The World By Lucky Dube',
        'Hand Of God',
        'morningInAmerica',
        'theGoodInMe',
        'wokeTheFCkUp',
        'Maybe Idk',
        'fashion',
        'He Is The Same',
        'youngDumbBrokeByKhalid',
        'eyoByAsa',
        'Jailer By Asa',
        'bestOfLuckyDube',
        'dottedLineJujuManByLabrinth',
        'Earthquake By Labrinth',
        'miracleByLabrinth',
        'No Ordinary By Labrinth',
        'oblivionByLabrinth',
        'Sexy Mf By Labrinth',
        'somethingsGotToGiveByLabrinth',
        'theProducerByLabrinth',
        'Thunderclouds By Lsd',
        'audioByLsd',
        'gunsRosesByLuckyDube',
        'Is This Freedom By Lucky Dube',
        'iveGotYouBabeByLuckyDube',
        'Love Me The Way Iam By Lucky Dube',
        'moneyMoneyMoneyByLuckyDube',
        'prisonerByLuckyDube',
        'Respect By Lucky Dube',
        'sleepingDogsByLuckyDube',
        'Fugitive By Lucky Dube',
        'differentColoursByLuckyDube',
        'feelIrieByLuckyDube',
        'Back To My Roots By Lucky Dube',
        'crazyWorldByLuckyDube',
        'Ding Ding Licky Licky Licky Bong By Lucky Dube',
        'goodGirlByLuckyDube',
        'houseOfExileByLuckyDube',
        'Its Not Easy By Lucky Dube',
        'loversInADangerousTimeByLuckyDube',
        'Remember Me By Lucky Dube',
        'slaveByLuckyDube',
        'taxmanByLuckyDube',
        'The Way It Is By Lucky Dube',
        'warAndCrimeByLuckyDube',
        'Majesty By Nicki Minaj And Eminem',
        'bestOfChrisBrown',
        'runIt',
        'yo(excuseMeMiss)',
        'gimmeThat',
        'sayGoodbye',
        'Kiss Kiss',
        'With You',
        'forever',
        'Deuces',
        'Look At Me Now',
        'yeah3x',
        'sheAintYou',
        'dontJudgeMe',
        'Fine China',
        'loyal',
        'New Flame',
        'noGuidance',
        'goCrazy',
        'Under The Influence',
        'residuals',
        'fallin',
        'ye',
        'Last Last',
        'onTheLow',
        'City Boys',
        'itsPlenty',
        'daiDai',
        'Dont Let Me Drown',
        'love',
        'Tested Approved Trusted',
        'gbona',
        'duduke',
        'Joromi',
        'tiff',
        'Love Dont Care',
        'smileForMe',
        'Where You Dey',
        'dayByDay',
        'selense',
        'jericho',
        'Ayo',
        'essenceByWizkidFtTems',
        'Holla At Your Boy',
        'Ojuelegba',
        'smileByWizkid',
        'Ginger',
        'Blessed',
        'moodByWizkid',
        'badToMe',
        'Money & Love',
        '2 Sugar',
        'togetherWithYou',
        'Free Mind',
        'higherByTems',
        'Damages',
        'Found',
        'Me & U',
        'Burning',
        'loveIsAKingdom',
        'Daily Blessings',
        'notAnAngel',
        'Ice T',
        'The Key',
        'Try Me',
        'Looku Looku',
        'Chandelier',
        'cheapThrills',
        'Elastic Heart',
        'Greatest',
        'Unstoppable',
        'Titanium',
        'Move Your Body',
        'aliveBySia',
        'Thundercloud',
        'Snowman',
        'Helium',
        'Bird Set Free',
        'Unavailable By Davido',
        'ifByDavido',
        'Fall',
        'FIA',
        'kanteByDavido',
        'Holy Ghost',
        'Soso By Omah Lay',
        'Understand',
        'badInfluence',
        'I Am',
        'Moving',
        'Rush By Ayra Starr',
        'Bloody Samaritan',
        'Sability',
        'Away',
        'Commando',
        'Calm Down By Rema',
        'Holiday',
        'Charm',
        'soundgasm',
        'Lonely At The Top By Asake',
        'Organise',
        'Amapiano',
        'Yoga',
        'Basquiat',
        '2:30',
        'Omo Ope',
        'Bandana By FireboyDML',
        'Peru',
        'Scatter',
        'Need You',
        'Jealous',
        'It Is What It Is',
        'highByAdekunleGold',
        'Sinner',
        'Okay',
        'Buga By Kizz Daniel',
        'Cough (Odo)',
        'Lie',
        'Pour Me Water',
        'Baby Riddim By Fave',
        'Kill Bill By SZA',
        'Snooze',
        'Shirt',
        'Good Days',
        'Saturn',
        'Location By Khalid',
        'Young Dumb & Broke',
        'Better',
        'Talk',
        'Coaster',
        'Mount Everest By Labrinth',
        'All For Us',
        "Still Don't Know My Name",
        'Never Felt So Alone',
        'Formula',
        'fireOnTheMountainByAsa',
        'Be My Man',
        'Moving On',
        'Abule By Patoranking',
        'Girlie O',
        'Confirm',
        'Love You Die',
        'Drogba (Joanna) By AfroB',
        'Monalisa By Lojay',
        'Ku Lo Sa By Oxlade',
        'People By Libianca',
        'Soweto By Victony',
        'reasonByOmahLay',
        'Coping Mechanism',
        'Water Spirit',
        'Bad',
        'Julia',
        'Amen',
        'Mary Go Round',
        'Waist',
        "Can't Help Myself",
        'Ozegba',
        'Doha',
        'Yebo',
        'Double',
        'Sooner',
        'Bahamas',
        'Winner',
        'Olabayo',
        'Sounds',
        'Jogodo',
        'Lost',
        'Worship',
        'Bundle By Bundle',
        'Fi Kan We Kan',
        'Metaverse',
        'Adenuga',
        'American Love',
        'Firegirl',
        'Funds',
        'One Call',
        'Angels',
        'Namek',
        '10 Toes',
        'Pronto',
        'Peaches Remix',
        'Pami',
        'Isaka II',
        'Artificial Happiness',
        'Clarity Of Mind',
        "Don't Call Me",
        'Beautiful Onyinye',
        'No One Like U',
        'Duffel Bag',
        'Safe Haven',
        'Sugarcane Remix',
        'Too Correct',
        'Wetin Be Love',
        'Maserati Remix',
        'Abeg',
        'Akanuche',
        'Life',
        'One Condition',
        'Dynamite By Tyla',
        'Hot Body',
        'Badman Gangsta',
        'Dopamine',
        'Success',
        'Congratulations',
        'Beatles',
        'the beatles',
        'abbey road',
        'let it be',
        'npr podcast',
        'npr news',
        'bbc radio',
        'spotify daily mix',
        'afrobeats playlist',
        'top hits',
        'chill playlist',
        'Miles davis',
        'kind of blue',
        'john coltrane',
        'led zeppelin',
        'pink floyd',
        'dark side of the moon',
        'michael jackson',
        'thriller',
        'bob marley',
        'legend',
        'fela kuti',
        'zombie',
        'water no get enemy',
        'lagbaja',
        'fuji music',
        'juju music',
        'mmsByAsakeWizkid',
        'Suru',
        'Skating',
        'Mentally',
        'Wave By AsakeCentralCee',
        'ActiveByAsake',
        'Fujuhouse',
        'I Swear',
        'Ligali',
        'Uhh Yeah',
        'M$NEY',
        'happinessByAsake',
        "What's Up My G",
        'Worldwide',
        'myHeartByAsake',
        'Checkmate',
        'Peace Be Unto You',
        'Terminator',
        'Dupe',
        'Muse',
        'Ototo',
        'Reason',
        'Sunmomi',
        'Baba God',
        'Joha',
        'Nzaza',
        'bornInTheWild',
        'Special Baby',
        'Wickedest',
        'Love Me Jeje',
        'Get It Right',
        'turnMeUp',
        'Boy O Boy',
        'What You Need',
        'T-Unit',
        'You In My Face',
        'Free Fall',
        'Voices In My Head',
        'Hold On',
        'Intermission',
        'Legacy',
        'Replay',
        'The Garden',
        'morningSunByTems',
        'Wait For U',
        'keseByWizkid',
        'Piece Of My Heart',
        'morayoByWizkid',
        'Bad Girl',
        'Everyday',
        'S2',
        'Diamonds',
        'Flower Pads',
        'specialByWizkid',
        'Ololufe',
        'Slip N Slide',
        'Plenty Loving',
        'Frames',
        'pongoByWizkid',
        'southGidi',
        'Easy With Me',
        'Nights In The Sun',
        'Turbulence',
        'Deep',
        'Sweet One',
        'Roma',
        'True Love',
        'Mighty Wine',
        'Longtime',
        'Cool Me Down',
        'gimmeLove',
        'danceAloneBySia',
        'Immortal Queen',
        'Little Wing',
        'One Night',
        'Go To Sleep',
        'Beautiful People',
        'awakeTonightBySia',
        'ranjhaBySia',
        'Street By Street',
        'Perfect',
        'I Forgive You',
        'Champion',
        'Hassle Free',
        'Washing Machine',
        'Everyday Is Christmas',
        'Candy Cane Lane',
        'Puppies Are Forever',
        'Sunshine',
        'Ho Ho Ho',
        "Santa's Coming",
        'saturnBySZA',
        'Ghost In The Machine',
        'F2F',
        'Notice Me',
        'Conceited',
        'Blind',
        'Seek & Destroy',
        'Special',
        'Broken Clocks',
        'Garden',
        'The Weekend',
        'Supermodel',
        'drewBarrymoreBySZA',
        'Normal Girl',
        'Prom',
        'Anything',
        'Pretty Little Birds',
        'openArmsBySZA',
        'Far',
        'Low',
        'I Hate U',
        'Sos',
        'Smoking On My Ex Pack',
        'Feel',
        'In The Garden',
        'Godfather',
        'Precision',
        'E No Finish',
        'naMoneyByDavido',
        'No Competition',
        'Picasso',
        'For The Road',
        'Jollof On The Jet',
        'commonPersonByBurnaBoy',
        'Higher',
        'Cheat On Me',
        'Virgil',
        'Big 7',
        'Alone',
        'Rollercoaster',
        'Whiskey',
        'Different Size',
        'Kilometre',
        'commasByAyraStarr',
        'Woman',
        'Good Luv',
        'Last Heartbreak Song',
        'The Year I Turned 21',
        'badVibesByAyraStarr',
        'Birds Sing',
        '21',
        'Control',
        'Water By Tyla',
        'Truth Or Dare',
        'Safer',
        'Butterflies',
        'On My Body',
        'Jump',
        'Art',
        'Declan Rice',
        'Dog Eat Dog',
        'picantoByOdumodu',
        'Firegun',
        'Blood On The Dance Floor',
        'Commend',
        'Saint Obi',
        'mountEverest',
        'Kill For Your Love',
        "I'm Tired",
        'Endorphins',
        'Power Coup',
        'Jungle',
        'Miracle',
        'Genius',
        'Audio',
        'Thunderclouds',
        'Mountains',
        'No New Friends',
        'heavenCanWaitByLSD',
        'Welcome To The Wonderful World Of',
        "It's Time",
        'Angel',
        'Reggae Strong',
        'Together As One',
        'Slave',
        'Prisoner',
        'The Way It Is',
        'House of Exile',
        'Crazy World',
        'Remember Me',
        'Going Back To My Roots',
        'Feel Irie',
        'Crime and Corruption',
        'God Bless The Women',
        'Different Colours',
        'One People',
        'Teach The World',
        'Truth in the World',
        'Shakara',
        'Lady',
        'Gentleman',
        "Teacher Don't Teach Me Nonsense",
        'Expensive Shit',
        'Roforofo Fight',
        'Opposite People',
        'Sorrow Tears and Blood',
        'Confusion',
        'Konko Below',
        'Gra Gra',
        'Nothing For You',
        'Suru Lere',
        'Sweet Mother',
        'Love Nwantiti',
        'location',
        'Eighteen',
        'American Teen',
        'Saved',
        'keepMeByKhalid',
        'Sun City',
        'Saturday Nights',
        'Vertigo',
        'Bluffin',
        'Outta My Head',
        'Up All Night',
        'Free Spirit',
        'Bad Luck',
        'Right Back',
        'Self',
        'Paradise',
        "I'm A Mess",
        'Woman By Rema',
        'Bounce',
        'Addicted',
        'Ginger Me',
        'Iron Man',
        'Dumebi',
        'Corny',
        'Bad Commando',
        'Aje',
        'Assurance',
        'If',
        'Flora My Flawa',
        'Mind',
        "Aww By Di'ja",
        'Osinachi',
        'Reggae Blues',
        'Pick Up By Adekunle Gold',
        'Ready',
        'Ariwo Ko',
        'Sina Rambo',
        'Gallardo',
        'Fans Mi',
        'The Sound',
        'Tchelete',
        'Aye',
        'Gobe',
        'Dami Duro',
        'Back When',
        'Over Dem',
        'Kante',
        'Na Money',
        'LCND',
        'Champions Sound',
        'Buga',
        'Cough',
        'My G',
        'Shu-Peru',
        'Feran Mi',
        'Dozie',
        'RTID',
        'Odo',
        'Eh God',
        'Barnabas',
        'Oshe',
        'Somebody Baby',
        'Flex',
        'Currently',
        'Nek-Unek',
        'Limpopo',
        'Pull Over',
        'Eledumare',
        'Double Wahala',
        'Show Me The Money',
     ]

LIGHT_TYPOS = ["ligh", "ligt", "lites", "ligths", "lite", "l1ghts"]
DOOR_TYPOS  = ["dors", "doores", "dores", "dorrs", "doos", "d0ors"]

# ── Relative increment phrases ────────────────────────────────────────
THERM_INCREMENT_PHRASES = [
    "Increase the temp by {n}.",
    "Turn it up by {n} degrees.",
    "Raise the temperature by {n}.",
    "Bump it up {n} degrees.",
    "Add {n} degrees.",
    "Make it {n} degrees warmer.",
    "Increase by {n}.",
    "Up the temp by {n}.",
]
THERM_DECREMENT_PHRASES = [
    "Decrease the temp by {n}.",
    "Turn it down by {n} degrees.",
    "Lower the temperature by {n}.",
    "Drop it {n} degrees.",
    "Reduce by {n}.",
    "Make it {n} degrees cooler.",
    "Decrease by {n}.",
    "Down the temp by {n}.",
]
THERM_VAGUE_WARMER = [
    "Make it a bit warmer.", "A little warmer please.",
    "Slightly warmer.", "Can you warm it up a bit?", "Warm it up slightly.",
]
THERM_VAGUE_COOLER = [
    "Make it a bit cooler.", "A little cooler please.",
    "Slightly cooler.", "Can you cool it down a bit?", "Cool it down slightly.",
]
THERM_OUT_OF_RANGE_PHRASES = [
    "Set the temp to {v}.", "Thermostat to {v}.",
    "Make it {v} degrees.", "Set it to {v}.",
    "I want it at {v} degrees.", "Set the temperature to {v}.",
]


def typo_word(word, variants, prob=0.10):
    return random.choice(variants) if random.random() < prob else word

def apply_typo(text, prob=0.15):
    # Only attempt typos based on probability
    if random.random() > prob:
        return text

    # Case-insensitive replacement for lights
    if re.search(r'lights?', text, re.IGNORECASE):
        typo = random.choice(LIGHT_TYPOS)
        # re.sub with IGNORECASE handles "Lights" or "lights"
        text = re.sub(r'lights?', typo, text, flags=re.IGNORECASE)
    
    # Case-insensitive replacement for doors
    if re.search(r'doors?', text, re.IGNORECASE):
        typo = random.choice(DOOR_TYPOS)
        # re.sub with IGNORECASE handles "Doors" or "doors"
        text = re.sub(r'doors?', typo, text, flags=re.IGNORECASE)
        
    return text

# ══════════════════════════════════════════════════════════════════════
# TOOL SCHEMA / SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════

def build_system_prompt(avail_r, avail_d):
    tv_rooms  = [r for r in avail_r if r in TV_ROOMS]
    spk_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
    fan_rooms = [r for r in avail_r if r in FAN_ROOMS]
    return (
        "You are a smart home assistant AI. Use tools to control the home.\n\n"
        "Output function calls as JSON.\n\n"
        "TOOLS:\n"
        "  toggle_lights(room, state='on'|'off')\n"
        "  lock_door(door, state='lock'|'unlock')\n"
        "  set_thermostat(temperature=<int 60–80>, mode='heat'|'cool'|'auto')\n"
        "  set_scene(scene='movie_night'|'bedtime'|'morning'|'away'|'party')\n"
        "  control_tv(room, state='on'|'off')\n"
        "  control_fan(room, state='on'|'off'[, speed='low'|'medium'|'high'])\n"
        "  control_speaker(room, action='play'|'pause'|'stop'|'next'|'previous'[, media='<str>'])\n"
        "  intent_unclear(reason='off_topic'|'incomplete'|'unsupported_device'"
        "|'unsupported_feature')\n\n"
        f"CONNECTED ROOMS (lights): {', '.join(avail_r)}\n"
        f"CONNECTED DOORS: {', '.join(avail_d)}\n"
        f"CONNECTED TVs: {', '.join(tv_rooms) if tv_rooms else 'none'}\n"
        f"CONNECTED SPEAKERS: {', '.join(spk_rooms) if spk_rooms else 'none'}\n"
        f"CONNECTED FANS: {', '.join(fan_rooms) if fan_rooms else 'none'}\n\n"
        "STATE RULES:\n"
        "  [STATE:] shows all current device states.\n"
        "  State already matches request → plain text reply, NO tool call.\n"
        "  Only rooms listed under CONNECTED TVs/SPEAKERS/FANS have those devices.\n"
        "  Requesting a device in an unlisted room → intent_unclear(unsupported_device).\n\n"
        "TV / SPEAKER / FAN RESOLUTION when user says 'the TV'/'the fan'/'the speaker':\n"
        "  1. Exactly one connected → use that room automatically.\n"
        "  2. Multiple connected + current_user_room has device → use current_user_room.\n"
        "  3. Multiple connected + exactly ONE is in the eligible state for the action\n"
        "     (e.g. only one TV is on and user says 'turn off the TV') → infer that room.\n"
        "  4. Multiple connected + ambiguous (rule 2 & 3 don't apply) "
        "→ intent_unclear(incomplete).\n\n"
        "LIGHT / DOOR RESOLUTION:\n"
        "  current_user_room set + connected → use current_user_room.\n"
        "  current_user_room set + NOT connected → intent_unclear(unsupported_device).\n"
        "  current_user_room empty → intent_unclear(incomplete).\n\n"
        "  [RECENT ACTIONS:] → transaction log, newest entry first. Format:\n"
        "    (X mins ago) [call1, call2, ...] -> summary.\n"
        "  Each [...] bracket is ONE command the user previously issued.\n"
        "  For 'undo'/'reverse'/'back': invert ONLY the most recent transaction\n"
        "    (the FIRST [...] block). Older transactions are always ignored.\n"
        "  For pronouns ('it'/'them'): refer to the device(s) in the first [...] block.\n"
        "  Do NOT use recent actions to infer which room 'the/this light' or 'the/this door'\n"
        "  refers to when current_user_room is explicitly set — current_user_room wins.\n"
        "  For 'all lights' / 'all doors': check STATE for each device — act only\n"
        "  on those whose state contradicts the request (issue individual tool calls).\n"
        "  SYNONYMS: 'open'='unlock'; 'close'/'shut'='lock'; 'skip'='next';\n"
        "  'back'='previous' (for speaker track navigation), but can also mean 'undo' for\n"
        "  reverting device states based on [RECENT ACTIONS].\n"
        "  'continue'/'resume'/'on the music'='play'; 'play <song/artist>' = action='play' + media='<str>'.\n"
        "  Relative state clauses ('the light that is on', 'the door that is locked')\n"
        "  override current_user_room — check STATE and act on the matching device."
    )

# ══════════════════════════════════════════════════════════════════════
# STATE HELPERS
# ══════════════════════════════════════════════════════════════════════

def sample_topology(required_rooms=None, required_doors=None, min_rooms=2, min_doors=2):
    req_r = required_rooms or []
    req_d = required_doors or []
    n_r   = random.randint(max(min_rooms, len(req_r)), len(ALL_ROOMS))
    pool_r = [r for r in ALL_ROOMS if r not in req_r]
    avail_r = req_r + random.sample(pool_r, min(n_r - len(req_r), len(pool_r)))
    random.shuffle(avail_r)
    n_d   = random.randint(max(min_doors, len(req_d)), len(ALL_DOORS))
    pool_d = [d for d in ALL_DOORS if d not in req_d]
    avail_d = req_d + random.sample(pool_d, min(n_d - len(req_d), len(pool_d)))
    random.shuffle(avail_d)
    return avail_r, avail_d

def generate_random_state(avail_r, avail_d):
    if random.random() < 0.25:
        ul = random.choice(["on", "off"])
        ud = random.choice(["locked", "unlocked"])
        return {
            "lights":   {r: {"state": ul} for r in avail_r},
            "doors":    {d: ud for d in avail_d},
            "thermostat": {"temperature": random.randint(MIN_T, MAX_T), "mode": random.choice(["heat", "cool", "auto"])},
            "active_scene": None,
            "tv":       {r: "off" for r in avail_r if r in TV_ROOMS},
            "speaker":  {r: "stopped" for r in avail_r if r in SPEAKER_ROOMS},
            "fan":      {r: {"state": "off", "speed": "low"} for r in avail_r if r in FAN_ROOMS},
        }
    # Original behavior
    return {
        "lights":   {r: {"state": random.choice(["on", "off"])} for r in avail_r},
        "doors":    {d: random.choice(["locked", "unlocked"]) for d in avail_d},
        "thermostat": {"temperature": random.randint(MIN_T, MAX_T), "mode": random.choice(["heat", "cool", "auto"])},
        "active_scene": None,
        "tv":       {r: random.choice(["on", "off"]) for r in avail_r if r in TV_ROOMS},
        "speaker":  {r: random.choice(["playing", "paused", "stopped"]) for r in avail_r if r in SPEAKER_ROOMS},
        "fan":      {r: {"state": random.choice(["on", "off"]), "speed": random.choice(["low", "medium", "high"])} for r in avail_r if r in FAN_ROOMS},
    }

def apply_force(state, force, avail_r, avail_d):
    if not force: return
    for k, v in force.get("lights", {}).items():
        if k in avail_r: state["lights"][k]["state"] = v
    for k, v in force.get("doors", {}).items():
        if k in avail_d: state["doors"][k] = v
    for k, v in force.get("tv", {}).items():
        if k in state.get("tv", {}): state["tv"][k] = v
    for k, v in force.get("speaker", {}).items():
        if k in state.get("speaker", {}): state["speaker"][k] = v
    for k, v in force.get("fan", {}).items():
        if k in state.get("fan", {}):
            if isinstance(v, dict): state["fan"][k].update(v)
            else: state["fan"][k]["state"] = v
    if "thermostat" in force: state["thermostat"] = force["thermostat"]

def format_state(state, user_room, avail_r, avail_d):
    lights  = ", ".join(f"{r}:{state['lights'][r]['state']}" for r in sorted(avail_r))
    doors   = ", ".join(f"{d}:{state['doors'][d]}" for d in sorted(avail_d))
    tv_d    = state.get("tv", {})
    sp_d    = state.get("speaker", {})
    fan_d   = state.get("fan", {})
    tv_str  = ", ".join(f"{r}:{tv_d[r]}" for r in sorted(tv_d))
    sp_str  = ", ".join(f"{r}:{sp_d[r]}" for r in sorted(sp_d))
    fan_str = ", ".join(
        f"{r}:{fan_d[r]['state']}({fan_d[r]['speed']})" for r in sorted(fan_d))
    therm  = state["thermostat"]
    scene  = state.get("active_scene") or "none"
    return (
        f"[STATE: lights={{{lights}}}, doors={{{doors}}}, "
        f"thermostat={therm['temperature']}F/{therm['mode']}, scene={scene}, "
        f"tv={{{tv_str}}}, speaker={{{sp_str}}}, fan={{{fan_str}}}, "
        f"current_user_room={user_room}]"
    )

def augment_text(text):
    if random.random() > 0.5: return text
    if random.random() < 0.3: text = text.lower()
    if random.random() < 0.3:
        pre = ["umm ", "hey, ", "can you ", "could you ", "please ", "just ",
               "i need you to ", "would you "]
        text = random.choice(pre) + text[0].lower() + text[1:]
    if random.random() < 0.2:
        sfx = [" please", " thanks", " right now", " asap", " as well", " too"]
        text = text.rstrip(".!?") + random.choice(sfx)
        if random.random() < 0.5: text += "."
    if random.random() < 0.2: text = text.replace(", ", ",")
    if random.random() < 0.4: text = text.rstrip(".!?")
    return text

# ══════════════════════════════════════════════════════════════════════
# v10 TRANSACTION LOG HELPERS
# ══════════════════════════════════════════════════════════════════════

def fmt_txn(mins_ago: int, call_strs: list, summary: str) -> str:
    t     = f"({mins_ago} min{'s' if mins_ago > 1 else ''} ago)"
    calls = ", ".join(call_strs)
    return f"{t} [{calls}] -> {summary}"

def rand_distractor_txn(avail_r: list, avail_d: list, mins_ago: int) -> str:
    """
    Return one realistic successful transaction log entry.
    Weights reflect the actual distribution of device types in production.
    """
    types   = ["light", "door", "compound", "thermostat", "scene", "tv", "speaker"]
    weights = [35,      20,     20,         10,           5,       4,    4      ]
 
    if any(r in FAN_ROOMS for r in avail_r):
        types.append("fan")
        weights.append(2)
 
    choice = random.choices(types, weights=weights)[0]
 
    # ── light ─────────────────────────────────────────────────────────────
    if choice == "light" and avail_r:
        r = random.choice(avail_r)
        s = random.choice(["on", "off"])
        return fmt_txn(mins_ago,
                       [f"toggle_lights(room={r}, state={s})"],
                       f"{ROOM_DISPLAY[r]} light turned {s}.")
 
    # ── door ──────────────────────────────────────────────────────────────
    if choice == "door" and avail_d:
        d = random.choice(avail_d)
        s = random.choice(["lock", "unlock"])
        aw = "locked" if s == "lock" else "unlocked"
        return fmt_txn(mins_ago,
                       [f"lock_door(door={d}, state={s})"],
                       f"{DOOR_DISPLAY[d]} {aw}.")
 
    # ── compound (mixed light + door in one block) ─────────────────────────
    # Matches the pattern where a previous turn successfully controlled
    # multiple device types together (e.g. lights + doors in one command).
    if choice == "compound" and avail_r and avail_d:
        n_lights = random.randint(1, min(3, len(avail_r)))
        n_doors  = random.randint(1, min(2, len(avail_d)))
        rooms    = random.sample(avail_r, n_lights)
        doors    = random.sample(avail_d, n_doors)
        ls       = random.choice(["on", "off"])
        ds       = random.choice(["lock", "unlock"])
        aw       = "locked" if ds == "lock" else "unlocked"
        call_strs = (
            [f"toggle_lights(room={r}, state={ls})" for r in rooms]
            + [f"lock_door(door={d}, state={ds})" for d in doors]
        )
        light_summary = " ".join(f"{ROOM_DISPLAY[r]} light {ls}." for r in rooms)
        door_summary  = " ".join(f"{DOOR_DISPLAY[d]} {aw}." for d in doors)
        return fmt_txn(mins_ago, call_strs, f"{light_summary} {door_summary}")
 
    # ── thermostat ────────────────────────────────────────────────────────
    if choice == "thermostat":
        t    = random.randint(MIN_T, MAX_T)
        mode = random.choice(["heat", "cool", "auto"])
        return fmt_txn(mins_ago,
                       [f"set_thermostat(temperature={t}, mode={mode})"],
                       f"Thermostat set to {t}F in {mode} mode.")
 
    # ── scene ─────────────────────────────────────────────────────────────
    if choice == "scene":
        sc = random.choice(SCENES)
        return fmt_txn(mins_ago,
                       [f"set_scene(scene={sc})"],
                       f"{sc.replace('_', ' ').title()} scene activated.")
 
    # ── tv ────────────────────────────────────────────────────────────────
    if choice == "tv" and any(r in TV_ROOMS for r in avail_r):
        r = random.choice([x for x in avail_r if x in TV_ROOMS])
        s = random.choice(["on", "off"])
        return fmt_txn(mins_ago,
                       [f"control_tv(room={r}, state={s})"],
                       f"{ROOM_DISPLAY[r]} TV turned {s}.")
 
    # ── speaker ───────────────────────────────────────────────────────────
    if choice == "speaker" and any(r in SPEAKER_ROOMS for r in avail_r):
        r = random.choice([x for x in avail_r if x in SPEAKER_ROOMS])
        a = random.choices(["play", "pause", "stop"], weights=[50, 25, 25])[0]
        j = random.choice(['act','not','act','act'])
        if a == "play" and j =='act':
            media   = random.choice(LOCAL_MUSIC)
            call    = f"control_speaker(room={r}, action=play, media={media})"
            summary = f"Playing '{media}' on the {ROOM_DISPLAY[r]} speaker."
        elif a == "play" and j =='not':
            call    = f"control_speaker(room={r}, action=play)"
            summary = f"{ROOM_DISPLAY[r]} speaker played."
        elif a == "pause":
            call    = f"control_speaker(room={r}, action=pause)"
            summary = f"{ROOM_DISPLAY[r]} speaker paused."
        else:
            call    = f"control_speaker(room={r}, action=stop)"
            summary = f"Stopped the music on the {ROOM_DISPLAY[r]} speaker."
        return fmt_txn(mins_ago, [call], summary)
 
    # ── fan ───────────────────────────────────────────────────────────────
    if choice == "fan" and any(r in FAN_ROOMS for r in avail_r):
        r = random.choice([x for x in avail_r if x in FAN_ROOMS])
        s = random.choice(["on", "off"])
        return fmt_txn(mins_ago,
                       [f"control_fan(room={r}, state={s})"],
                       f"{ROOM_DISPLAY[r]} fan turned {s}.")
 
    # ── fallback: light ───────────────────────────────────────────────────
    if avail_r:
        r = random.choice(avail_r)
        s = random.choice(["on", "off"])
        return fmt_txn(mins_ago,
                       [f"toggle_lights(room={r}, state={s})"],
                       f"{ROOM_DISPLAY[r]} light turned {s}.")
 
    t    = random.randint(MIN_T, MAX_T)
    mode = random.choice(["heat", "cool", "auto"])
    return fmt_txn(mins_ago,
                   [f"set_thermostat(temperature={t}, mode={mode})"],
                   f"Thermostat set to {t}F in {mode} mode.")
 
 
def build_distractor_log(avail_r, avail_d, n=None, start_mins=None):
    """
    Build N distractor transaction log entries with realistic timestamps.
 
    start_mins defaults to 1–6 (was 8–18).
    Gap between consecutive entries is 0–5 mins (was 8–20).
    Zero-gap (same-minute entries) occurs ~15% of the time, matching
    the production pattern of rapid back-to-back turns.
 
    Call-site overrides still work: build_distractor_log(..., start_mins=3)
    """
    if n is None:
        n = random.randint(1, 3)
    if start_mins is None:
        start_mins = random.randint(1, 16)
 
    entries, mins = [], start_mins
    for _ in range(n):
        entries.append(rand_distractor_txn(avail_r, avail_d, mins))
        gap  = 0 if random.random() < 0.15 else random.randint(1, 15)
        mins += gap
 
    return "\n".join(entries)

def get_cross_room_txn(avail_r, avail_d, room_to_avoid, mins_ago=None):
    if mins_ago is None: mins_ago = random.randint(5, 20)
    others = [r for r in avail_r if r != room_to_avoid]
    if not others:
        if avail_d:
            d = random.choice(avail_d); s = random.choice(["lock", "unlock"])
            aw = "locked" if s == "lock" else "unlocked"
            return fmt_txn(mins_ago, [f"lock_door(door={d}, state={s})"],
                           f"{DOOR_DISPLAY[d]} {aw}.")
        t = random.randint(MIN_T, MAX_T)
        return fmt_txn(mins_ago, [f"set_thermostat(temperature={t}, mode=auto)"],
                       f"Thermostat set to {t}F.")
    other = random.choice(others); s = random.choice(["on", "off"])
    return fmt_txn(mins_ago, [f"toggle_lights(room={other}, state={s})"],
                   f"{ROOM_DISPLAY[other]} light turned {s}.")

# ══════════════════════════════════════════════════════════════════════
# BUILD EXAMPLE
# FIX-D: Auto-append ACTION REQUIRED / ACTION NOT REQUIRED to think traces
# ══════════════════════════════════════════════════════════════════════

def build_ex(
    user_prompt, calls_data, response_text, avail_r, avail_d, state,
    user_room=None, action_log=None, think_trace="", category="", augment=True
):
    # SYSTEMIC FIX 1: Dynamic Production Location Noise (70% in a room, 30% empty)
    if user_room is None:
        user_room = random.choice(avail_r) if random.random() < 0.7 else ""

    # SYSTEMIC FIX 2: Universal Background Log Noise (85% chance of logs)
    if action_log is None:
        action_log = build_distractor_log(avail_r, avail_d) if random.random() < 0.85 else ""

    prompt = augment_text(user_prompt) if augment else user_prompt
    state_str   = format_state(state, user_room, avail_r, avail_d)
    full_prompt = state_str
    if action_log: full_prompt += f"\n[RECENT ACTIONS:\n{action_log}]"
    full_prompt += f"\n{prompt}"
 
    if think_trace:
        stripped = think_trace.rstrip()
        marker_action    = "ACTION REQUIRED."
        marker_no_action = "ACTION NOT REQUIRED. Text reply only."
        already_marked = (stripped.endswith(marker_action)
                          or stripped.endswith(marker_no_action))
        if not already_marked:
            think_trace = stripped + (
                f" {marker_action}" if calls_data else f" {marker_no_action}"
            )
 
        if calls_data and len(calls_data) >= 2 and "Total:" not in think_trace:
            n_c = len(calls_data)
            call_word = "call" if n_c == 1 else "calls"
            think_trace = re.sub(
                r'\s*ACTION REQUIRED\.$',
                f' Total: {n_c} tool {call_word} required. Emitting all {n_c}. ACTION REQUIRED.',
                think_trace,
            )
 
    content_str = f"<think>{think_trace}</think>\n" if think_trace else ""
    if calls_data:
        for d in calls_data:
            tool_json = json.dumps({"name": d["name"], "parameters": d["args"]})
            content_str += f"<|tool_call_start|>{tool_json}<|tool_call_end|>\n"
    content_str += response_text
 
    messages = [
        {"role": "system",    "content": build_system_prompt(avail_r, avail_d)},
        {"role": "user",      "content": full_prompt},
        {"role": "assistant", "content": content_str.strip()}
    ]
    return {"messages": messages, "category": category, "source": "stage2_v14"}
# ══════════════════════════════════════════════════════════════════════
# NEW v12: HYBRID EXHAUSTIVE GENERATORS (unchanged from v11)
# ══════════════════════════════════════════════════════════════════════

def gen_exhaustive_light_logic(synonyms_per_combo: int = 25) -> list:
    """
    HYBRID EXHAUSTIVE — Light control.
    Logic axes (16 combinations): request × init_state × user_room × log_type
    Total: 16 × 25 = 400 examples.
    """
    examples = []
    ON_PHRASES  = [
        "Turn the light on.", "Lights on.", "On the light.", "On this light.",
        "Switch on the light.", "Light on please.", "It's dark in here.",
        "Turn on the light.", "Can you turn on the light?", "I can't see.",
        "Lights on please.", "Switch on.", "Put the light on.",
    ]
    OFF_PHRASES = [
        "Turn the light off.", "Lights off.", "Off the light.", "Off this light.",
        "Switch off the light.", "Light off please.", "Too bright.",
        "Turn off the light.", "Kill the light.", "Lights off please.",
        "Cut the light.", "Switch off.", "Kill the lights in here.",
    ]

    for req, init, user_room_type, log_type in itertools.product(
        ["on", "off"], ["on", "off"], ["connected", "empty"], ["none", "cross_room"]
    ):
        opp = "off" if req == "on" else "on"
        for _ in range(synonyms_per_combo):
            r       = random.choice(ALL_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[r])
            state   = generate_random_state(avail_r, avail_d)
            apply_force(state, {"lights": {r: init}}, avail_r, avail_d)
            alias     = random.choice(ROOM_ALIASES[r])
            user_room = r if user_room_type == "connected" else ""
            action_log = ""
            if log_type == "cross_room" and len(avail_r) > 1:
                action_log = get_cross_room_txn(avail_r, avail_d, r)
            prompt = random.choice(ON_PHRASES if req == "on" else OFF_PHRASES)

            if user_room_type == "empty":
                think = (
                    f"User said '{prompt}'. "
                    f"current_user_room is empty. "
                    f"Need a specific room to control the light. "
                    f"Calling intent_unclear(incomplete)."
                )
                examples.append(build_ex(prompt,
                    [{"name": "intent_unclear", "args": {"reason": "incomplete"}}],
                    "Which room's light would you like me to control?",
                    avail_r, avail_d, state,
                    user_room="", action_log=action_log,
                    think_trace=think, category="exhaustive_light_logic"))

            elif req == init:
                
                think = (
                    f"User wants {r} light {req}. "
                    f"STATE shows {r}:{init}. Already matches. "
                    f"No tool call needed."
                )
                examples.append(build_ex(prompt, [],
                    f"The {alias} light is already {req}.",
                    avail_r, avail_d, state,
                    user_room=user_room, action_log=action_log,
                    think_trace=think, category="exhaustive_light_logic"))

            else:
                
                think = (
                    f"User is in '{r}'. User said '{prompt}'. "
                    f"Current state is {opp}, user wants {req}. "
                    f"Calling toggle_lights(room={r}, state={req})."
                )
                examples.append(build_ex(prompt,
                    [{"name": "toggle_lights", "args": {"room": r, "state": req}}],
                    f"The {alias} light is now {req}.",
                    avail_r, avail_d, state,
                    user_room=user_room, action_log=action_log,
                    think_trace=think, category="exhaustive_light_logic"))

    return examples

def gen_gadget_explicit_room_rejection(target: int = 1_500) -> list:
    """Fixes bug where Gadget Rule 1 overrides an explicitly named wrong room."""
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        
        device = random.choice(["tv", "speaker", "fan"])
        valid_rooms = TV_ROOMS if device == "tv" else (SPEAKER_ROOMS if device == "speaker" else FAN_ROOMS)
        
        # Ensure we have at least 1 valid connected device
        conn = [r for r in avail_r if r in valid_rooms]
        if not conn: continue
        
        # Pick an explicit room that does NOT have the device
        wrong_rooms = [r for r in avail_r if r not in valid_rooms]
        if not wrong_rooms: continue
        wrong_room = random.choice(wrong_rooms)
        wrong_alias = random.choice(ROOM_ALIASES[wrong_room])
        
        prompt = f"Turn on the {wrong_alias} {device}."
        if device == "speaker":
            comm = random.choice(LOCAL_MUSIC)
            k = random.choice(['Hit','Play','Give me','I want to hear','Lets play','Play for me'])
            prompt = f"{k} {comm} in the {wrong_alias}."
        
        conn_str = ", ".join(conn)
        think = (
            f"User explicitly asked to control the {device} in '{wrong_room}'. "
            f"Checking CONNECTED {device.upper()}S: [{conn_str}]. "
            f"'{wrong_room}' is NOT in the list. "
            f"Calling intent_unclear(unsupported_device)."
        )
        
        examples.append(build_ex(prompt, 
            [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
            f"There's no {device} connected in the {wrong_alias.title()}.",
            avail_r, avail_d, state, think_trace=think, category="gadget_explicit_room_rejection"))
    return examples
    
def gen_current_room_unsupported_device(target: int = 600) -> list:
    """
    Trains the single valid implicit-local rejection case:
    user in a room with no connected door says 'the door'.
    
    TV/speaker/fan are intentionally excluded — implicit requests for
    those devices go through gadget resolution Rules 1-4, NOT rejection.
    Only DOOR uses current_user_room for resolution, so only doors can
    produce an unsupported_device result from an implicit request.
    
    (Explicit room requests like 'turn on the hallway TV' are handled
    in gen_rejections device_tv/device_fan/device_door branches.)
    """
    examples = []
    verbs = [
        "close the door", "open the door", "lock the door",
        "shut the door", "unlock the door", "open this door",
        "close this door", "lock this door",
    ]
    while len(examples) < target:
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        # Rooms whose name does not appear in avail_d (hallway is primary case)
        rooms_without_door = [r for r in avail_r if r not in avail_d]
        if not rooms_without_door:
            continue
        u_room = random.choice(rooms_without_door)
        verb   = random.choice(verbs)
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""
        think = (
            f"User is in '{u_room}'. Said '{verb.capitalize()}.'. "
            f"'the door'/'this door' resolves to current_user_room='{u_room}'. "
            f"Checking CONNECTED DOORS from system prompt: [{', '.join(avail_d)}]. "
            f"'{u_room}' is NOT in this list. "
            f"Calling intent_unclear(unsupported_device)."
        )
        examples.append(build_ex(f"{verb.capitalize()}.",
            [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
            f"There's no door connected in the {ROOM_DISPLAY[u_room]}.",
            avail_r, avail_d, state,
            user_room=u_room, action_log=action_log,
            think_trace=think, category="current_room_unsupported"))
    return examples

def gen_explicit_room_over_log(target: int = 2_000) -> list:
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3)
        state = generate_random_state(avail_r, avail_d)
        
        target_r = random.choice(avail_r)
        log_r = random.choice([r for r in avail_r if r != target_r])
        user_r = random.choice([r for r in avail_r if r not in [target_r, log_r]] + ["", target_r])
        
        s = random.choice(["on", "off"])
        opp = "off" if s == "on" else "on"
        apply_force(state, {"lights": {target_r: opp}}, avail_r, avail_d)
        
        log_mins = random.randint(1, 5)
        action_log = get_cross_room_txn(avail_r, avail_d, target_r, log_mins)
        action_log += "\n" + build_distractor_log(avail_r, avail_d, n=random.randint(1,2), start_mins=log_mins+2)
        
        alias = random.choice(ROOM_ALIASES[target_r])
        prompt = f"{'Turn on' if s=='on' else 'Turn off'} the {alias} light."
        
        think = (
            f"User explicitly named '{target_r}' in the command. "
            f"Target room is '{target_r}'. "
            f"Current state is '{opp}'. Target state is '{s}'. "
            f"Calling toggle_lights(room={target_r}, state={s})."
        )
        
        examples.append(build_ex(prompt, 
            [{"name": "toggle_lights", "args": {"room": target_r, "state": s}}],
            f"The {alias} light is now {s}.",
            avail_r, avail_d, state,
            user_room=user_r, action_log=action_log, think_trace=think, 
            category="explicit_room_over_log"))
    return examples
    

def gen_thermostat_incremental(target: int = 1_200) -> list:
    """
    Trains relative temperature adjustments ('increase by 2', 'turn it up 5')
    and out-of-range absolute requests.

    Three cases:
    A) Increment/decrement → result IN range  → call set_thermostat(new_val)
    B) Increment/decrement → result OUT of range → inform user, clamp to boundary
    C) Absolute request out of range (< 60 or > 80) → inform user of valid range
    """
    examples = []

    # ── Case A & B: relative adjustments ─────────────────────────────
    for _ in range(int(target * 0.6)):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        cur   = state["thermostat"]["temperature"]
        cur_mode = state["thermostat"]["mode"]
        direction = random.choice(["up", "down"])
        n = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 10])

        if direction == "up":
            new_val = cur + n
            prompt  = random.choice(THERM_INCREMENT_PHRASES).format(n=n)
        else:
            new_val = cur - n
            prompt  = random.choice(THERM_DECREMENT_PHRASES).format(n=n)

        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""
        u_room = random.choice(["", random.choice(avail_r)])

        if MIN_T <= new_val <= MAX_T:
            # Case A: in range — call the tool
            mode = "heat" if new_val > cur else "cool"
            think = (
                f"User wants to {direction} temperature by {n}°F. "
                f"Current: {cur}F. "
                f"New value: {cur} {'+ ' if direction == 'up' else '- '}{n} = {new_val}F. "
                f"{new_val}F is within the valid range [{MIN_T}–{MAX_T}]. "
                f"Calling set_thermostat(temperature={new_val}, mode='{mode}')."
            )
            examples.append(build_ex(prompt,
                [{"name": "set_thermostat", "args": {"temperature": new_val, "mode": mode}}],
                f"Thermostat set to {new_val}°F in {mode} mode.",
                avail_r, avail_d, state,
                user_room=u_room, action_log=action_log,
                think_trace=think, category="thermostat_incremental"))
        else:
            # Case B: result out of range — clamp and inform
            clamped = max(MIN_T, min(MAX_T, new_val))
            boundary_dir = "maximum" if new_val > MAX_T else "minimum"
            boundary_val = MAX_T if new_val > MAX_T else MIN_T
            mode = "heat" if clamped > cur else "cool" if clamped < cur else cur_mode
            think = (
                f"User wants to {direction} temperature by {n}°F. "
                f"Current: {cur}F. "
                f"Computed new value: {new_val}F. "
                f"{new_val}F exceeds the valid range [{MIN_T}–{MAX_T}]. "
                f"Clamping to the {boundary_dir} allowed value of {boundary_val}F. "
                f"Calling set_thermostat(temperature={clamped}, mode='{mode}')."
            )
            examples.append(build_ex(prompt,
                [{"name": "set_thermostat", "args": {"temperature": clamped, "mode": mode}}],
                f"That would go out of range — setting to the {boundary_dir} of {clamped}°F.",
                avail_r, avail_d, state,
                user_room=u_room, action_log=action_log,
                think_trace=think, category="thermostat_incremental"))

    # ── Vague warmer/cooler ───────────────────────────────────────────
    for _ in range(int(target * 0.15)):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        cur   = state["thermostat"]["temperature"]
        cur_mode = state["thermostat"]["mode"]
        direction = random.choice(["warmer", "cooler"])
        delta = random.choice([2, 3])
        prompt = random.choice(
            THERM_VAGUE_WARMER if direction == "warmer" else THERM_VAGUE_COOLER)
        new_val = cur + delta if direction == "warmer" else cur - delta
        new_val = max(MIN_T, min(MAX_T, new_val))
        mode = "heat" if new_val > cur else "cool" if new_val < cur else cur_mode
        u_room = random.choice(["", random.choice(avail_r)])
        think = (
            f"User wants it '{direction}'. Interpreting as +{delta}°F change. "
            f"Current: {cur}F. New value: {new_val}F (clamped to [{MIN_T}–{MAX_T}] if needed). "
            f"Calling set_thermostat(temperature={new_val}, mode='{mode}')."
        )
        examples.append(build_ex(prompt,
            [{"name": "set_thermostat", "args": {"temperature": new_val, "mode": mode}}],
            f"Thermostat adjusted to {new_val}°F in {mode} mode.",
            avail_r, avail_d, state,
            user_room=u_room, think_trace=think, category="thermostat_incremental"))

    # ── Case C: absolute out-of-range request ─────────────────────────
    for _ in range(int(target * 0.25)):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        # Generate a value clearly outside the valid range
        out_val = random.choice(
            list(range(30, MIN_T)) + list(range(MAX_T + 1, 110))
        )
        prompt = random.choice(THERM_OUT_OF_RANGE_PHRASES).format(v=out_val)
        u_room = random.choice(["", random.choice(avail_r)])
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.3 else ""
        think = (
            f"User wants thermostat at {out_val}F. "
            f"Valid range is [{MIN_T}–{MAX_T}]F. "
            f"{out_val}F is outside the allowed range. "
            f"No tool call — informing user of the valid range."
        )
        examples.append(build_ex(prompt, [],
            f"I can only set the thermostat between {MIN_T}°F and {MAX_T}°F. "
            f"Please choose a value in that range.",
            avail_r, avail_d, state,
            user_room=u_room, action_log=action_log,
            think_trace=think, category="thermostat_incremental"))

    return examples


def gen_list_and_local_device(target: int = 1_500) -> list:
    """
    Trains: list of named rooms + implicit local device.
    'On the kitchen and bedroom light and open this door.'
    Restricted to room-mapped doors so user_room is always a valid room name.
    """
    examples = []
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=3)
        state = generate_random_state(avail_r, avail_d)

        # Only use doors that share their name with a room (valid user_room)
        valid_local_doors = [d for d in avail_d if d in ROOM_DOOR_ROOMS]
        if not valid_local_doors:
            continue
        local_d = random.choice(valid_local_doors)

        n_light_rooms = random.randint(2, 3)
        light_rooms = random.sample(
            [r for r in avail_r if r != local_d], min(n_light_rooms, len(avail_r) - 1)
        )
        if not light_rooms:
            continue

        ls = random.choice(["on", "off"])
        opp_ls = "off" if ls == "on" else "on"
        ds = random.choice(["lock", "unlock"])
        aw = "locked" if ds == "lock" else "unlocked"
        opp_aw = "unlocked" if ds == "lock" else "locked"

        for r in light_rooms:
            apply_force(state, {"lights": {r: opp_ls}}, avail_r, avail_d)
        apply_force(state, {"doors": {local_d: opp_aw}}, avail_r, avail_d)

        l_verb = "on" if ls == "on" else "off"
        d_verb = random.choice(["open", "unlock"]) if ds == "unlock" \
                 else random.choice(["close", "lock"])
        light_aliases = [random.choice(ROOM_ALIASES[r]) for r in light_rooms]
        d_alias = random.choice(DOOR_ALIASES[local_d])

        if len(light_rooms) == 2:
            light_str = f"the {light_aliases[0]} and {light_aliases[1]} light"
        else:
            light_str = ("the " + ", the ".join(light_aliases[:-1])
                         + f", and {light_aliases[-1]} light")

        door_p = random.choice(["this door", "the door"])
        prompt = random.choice([
            f"{l_verb.capitalize()} {light_str} and {d_verb} {door_p}.",
            f"{d_verb.capitalize()} {door_p} and {l_verb} {light_str}.",
        ])

        calls = [{"name": "toggle_lights", "args": {"room": r, "state": ls}}
                 for r in light_rooms]
        calls.append({"name": "lock_door",
                      "args": {"door": local_d, "state": ds}})

        n_total = len(light_rooms) + 1
        call_traces = ", ".join(f"toggle_lights({r}, {ls})" for r in light_rooms)
        think = (
            f"Compound request. "
            f"Part 1: explicit light list ({', '.join(light_rooms)}) → "
            f"turn {ls}: {call_traces}. "
            f"Part 2: '{door_p}' — implicit local door — current_user_room='{local_d}' → "
            f"lock_door(door={local_d}, state={ds}). "
            f"Issuing exactly {n_total} tool calls."
        )
        resp = " ".join(f"{a.title()} light {ls}." for a in light_aliases) \
               + f" {d_alias.title()} {aw}."

        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=local_d, think_trace=think, category="list_and_local"))
    return examples

def gen_exhaustive_pronoun_states(synonyms_per_combo: int = 20) -> list:
    """
    HYBRID EXHAUSTIVE — Pronoun resolution.
    Logic axes (12 combinations): logged_action × user_room_type × prompt_direction
    Total: 12 × 20 = 240 examples.
    """
    examples = []
    OFF_PRONOUNS = ["Turn it off.", "Kill it.", "Shut it off.",
                    "Off it.", "Switch it off.", "Cut it."]
    ON_PRONOUNS  = ["Turn it on.", "Switch it on.", "Put it back on.",
                    "On it.", "Turn it back on."]

    for s_log, user_room_type in itertools.product(
        ["on", "off"], ["same_as_log", "different_from_log", "empty"]
    ):
        target_s = "off" if s_log == "on" else "on"
        for _ in range(synonyms_per_combo):
            r_log   = random.choice(ALL_ROOMS)
            r_other = random.choice([x for x in ALL_ROOMS if x != r_log])
            avail_r, avail_d = sample_topology(required_rooms=[r_log, r_other])
            state   = generate_random_state(avail_r, avail_d)
            apply_force(state, {"lights": {r_log: s_log}}, avail_r, avail_d)
            primary_mins = random.randint(1, 4)
            primary_txn  = fmt_txn(primary_mins,
                                   [f"toggle_lights(room={r_log}, state={s_log})"],
                                   f"{ROOM_DISPLAY[r_log]} light turned {s_log}.")
            if random.random() < 0.5:
                dist = build_distractor_log(avail_r, avail_d, n=1,
                                            start_mins=primary_mins + random.randint(7, 14))
                action_log = primary_txn + "\n" + dist
            else:
                action_log = primary_txn
            alias_log = random.choice(ROOM_ALIASES[r_log])
            prompt    = random.choice(OFF_PRONOUNS if target_s == "off" else ON_PRONOUNS)
            if user_room_type == "same_as_log":
                user_room = r_log
            elif user_room_type == "different_from_log":
                user_room = r_other
            else:
                user_room = ""
            t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
            think = (
                f"User said '{prompt}'. "
                f"The pronoun 'it' refers to the device in the first [...] block "
                f"({t_label}): {ROOM_DISPLAY[r_log]} light. "
                f"Current state of {r_log} is '{s_log}'. "
                f"User wants it '{target_s}' (opposite of logged state). "
                f" Pronouns resolve directly to the logged device. "
                f"Calling toggle_lights(room={r_log}, state={target_s})."
            )
            examples.append(build_ex(prompt,
                [{"name": "toggle_lights", "args": {"room": r_log, "state": target_s}}],
                f"The {alias_log} light is now {target_s}.",
                avail_r, avail_d, state,
                user_room=user_room, action_log=action_log,
                think_trace=think, category="exhaustive_pronoun_states"))

    return examples


def gen_exhaustive_gadget_rules(synonyms_per_combo: int = 20) -> list:
    """
    HYBRID EXHAUSTIVE — TV/Speaker/Fan resolution rules 1-4.
    ~400 examples covering 100% of gadget resolution decision tree.
    """
    examples = []
    TV_ON_T  = ["Turn on the TV.", "TV on.", "On the TV.", "Start the TV.", "Power the TV."]
    TV_OFF_T = ["Turn off the TV.", "TV off.", "Off the TV.", "Power off the TV."]
    SP_PLAY  = ["Play music.", "On the speaker.", "Start the music.", "Play some tunes."]
    SP_STOP  = ["Stop the music.", "Off the speaker.", "Stop the speaker.", "Kill the music."]
    FAN_ON_T = ["Turn on the fan.", "Fan on.", "On the fan.", "Switch on the fan."]
    FAN_OFF_T= ["Turn off the fan.", "Fan off.", "Off the fan.", "Switch off the fan."]

    device_configs = {
        "tv":      (TV_ROOMS,     TV_ON_T,  TV_OFF_T,  "tv",      "on",   "off"),
        "speaker": (SPEAKER_ROOMS,SP_PLAY,  SP_STOP,   "speaker", "play", "stop"),
        "fan":     (FAN_ROOMS,    FAN_ON_T, FAN_OFF_T, "fan",     "on",   "off"),
    }

    # Helper to generate the correct tool call and strings
    def build_gadget_action(dt, rm, act, al, med=None):
        if dt == "tv":
            call = {"name": "control_tv", "args": {"room": rm, "state": act}}
            t_str = f"control_tv(room={rm}, state={act})"
            rp = f"The {al} TV is now {act}."
        elif dt == "speaker":
            args = {"room": rm, "action": act}
            m_str = ""
            if act == "play" and med:
                args["media"] = med
                m_str = f", media='{med}'"
            call = {"name": "control_speaker", "args": args}
            t_str = f"control_speaker(room={rm}, action={act}{m_str})"
            if act == "play":
                rp = f"Playing '{med}' on the {al} speaker." if med else f"Playing music on the {al} speaker."
            else:
                rp = f"Stopped the music on the {al} speaker."
        else:
            call = {"name": "control_fan", "args": {"room": rm, "state": act}}
            t_str = f"control_fan(room={rm}, state={act})"
            rp = f"The {al} fan is now {act}."
        return call, t_str, rp

    for device_type, (dev_rooms, on_tmpl, off_tmpl, _, act_on, act_off) \
            in device_configs.items():
        for desired, tmpls in [(act_on, on_tmpl), (act_off, off_tmpl)]:
            for rule in ["rule1", "rule2", "rule3", "rule4"]:
                for _ in range(synonyms_per_combo):
                    avail_r, avail_d = sample_topology(
                        required_rooms=[random.choice(dev_rooms)])
                    connected = [r for r in avail_r if r in dev_rooms]
                    if not connected: continue
                    state = generate_random_state(avail_r, avail_d)
                    action_log = build_distractor_log(avail_r, avail_d, n=1) \
                        if random.random() < 0.4 else ""

                    # Dynamically handle media for speaker play commands
                    media_val = None
                    if device_type == "speaker" and desired == "play" and random.random() < 0.5:
                        media_val = random.choice(LOCAL_MUSIC)
                        prompt = random.choice([f"Play {media_val}.", f"Put on {media_val}.", f"Start {media_val}."])
                    else:
                        prompt = random.choice(tmpls)

                    if rule == "rule1":
                        avail_r_r1 = list(dict.fromkeys(
                            [connected[0]] + [r for r in avail_r if r not in dev_rooms]))
                        state_r1   = generate_random_state(avail_r_r1, avail_d)
                        r          = connected[0]
                        alias      = random.choice(ROOM_ALIASES[r])
                        
                        if device_type == "tv":
                            state_r1["tv"][r] = "off" if desired == "on" else "on"
                        elif device_type == "speaker":
                            state_r1["speaker"][r] = "stopped" if desired == "play" else "playing"
                        else:
                            state_r1["fan"][r]["state"] = "off" if desired == "on" else "on"

                        call, tool_call_str, resp = build_gadget_action(device_type, r, desired, alias, media_val)
                        
                        think = (
                            f"Checking CONNECTED {device_type.upper()}S: [{r}]. "
                            f"Exactly one {device_type} is connected. "
                            f"Resolving to {r}. "
                            f"Calling {tool_call_str}."
                        )
                        examples.append(build_ex(prompt, [call], resp,
                            avail_r_r1, avail_d, state_r1,
                            action_log=action_log,
                            think_trace=think, category="exhaustive_gadget_rules"))

                    elif rule == "rule2" and len(connected) >= 2:
                        r         = random.choice(connected)
                        alias     = random.choice(ROOM_ALIASES[r])
                        user_room = r
                        
                        if device_type == "tv":
                            state["tv"][r] = "off" if desired == "on" else "on"
                        elif device_type == "speaker":
                            state["speaker"][r] = "stopped" if desired == "play" else "playing"
                        else:
                            state["fan"][r]["state"] = "off" if desired == "on" else "on"

                        call, tool_call_str, resp = build_gadget_action(device_type, r, desired, alias, media_val)
                        conn_str = ", ".join(connected)
                        
                        think = (
                            f"Checking CONNECTED {device_type.upper()}S: [{conn_str}]. "
                            f"Multiple {device_type}s connected. "
                            f"current_user_room='{r}' has a {device_type}. "
                            f"Resolving to {r}. "
                            f"Calling {tool_call_str}."
                        )
                        examples.append(build_ex(prompt, [call], resp,
                            avail_r, avail_d, state,
                            user_room=user_room, action_log=action_log,
                            think_trace=think, category="exhaustive_gadget_rules"))

                    elif rule == "rule3" and len(connected) >= 2:
                        r_target = connected[0]
                        
                        # Force ALL other connected devices to be ineligible
                        for r_other in connected[1:]:
                            if device_type == "tv":
                                state["tv"][r_other] = "on" if desired == "on" else "off"
                            elif device_type == "speaker":
                                state["speaker"][r_other] = "playing" if desired in ["play", "resume"] else "stopped"
                            else:
                                state["fan"][r_other]["state"] = "on" if desired == "on" else "off"

                        alias = random.choice(ROOM_ALIASES[r_target])
                        if device_type == "tv":
                            state["tv"][r_target] = "off" if desired == "on" else "on"
                        elif device_type == "speaker":
                            state["speaker"][r_target] = "stopped" if desired == "play" else "playing"
                        else:
                            state["fan"][r_target]["state"] = "off" if desired == "on" else "on"

                        call, tool_call_str, resp = build_gadget_action(device_type, r_target, desired, alias, media_val)
                            
                        non_dev   = [r for r in avail_r if r not in connected]
                        user_room = random.choice(non_dev) if non_dev else ""
                        
                        conn_str = ", ".join(connected)
                        room_clause = (f"User is in '{user_room}' which has no {device_type}."
                                       if user_room else "User's location is unknown.")
                        
                        think = (
                            f"Checking CONNECTED {device_type.upper()}S: [{conn_str}]. "
                            f"Multiple {device_type}s connected. {room_clause} "
                            f"Checking states: exactly ONE {device_type} ({r_target}) is in the eligible state for '{desired}'. "
                            f"Inferring {r_target}. "
                            f"Calling {tool_call_str}."
                        )
                        examples.append(build_ex(prompt, [call], resp,
                            avail_r, avail_d, state,
                            user_room=user_room, action_log=action_log,
                            think_trace=think, category="exhaustive_gadget_rules"))

                    elif rule == "rule4" and len(connected) >= 2:
                        if device_type == "tv":
                            for r in connected:
                                state["tv"][r] = "off" if desired == "on" else "on"
                        elif device_type == "speaker":
                            e_state = "stopped" if desired == "play" else "playing"
                            for r in connected:
                                state["speaker"][r] = e_state
                        else:
                            for r in connected:
                                state["fan"][r]["state"] = "off" if desired == "on" else "on"
                        
                        non_dev   = [r for r in avail_r if r not in connected]
                        user_room = random.choice(non_dev) if non_dev else ""
                        clarify   = (f"Which {device_type}? I have them in: "
                                     f"{', '.join(ROOM_DISPLAY[r] for r in connected)}.")
                        
                        conn_str = ", ".join(connected)
                        room_clause = (f"User is in '{user_room}' which has no {device_type}."
                                       if user_room else "User's location is unknown.")
                        
                        think = (
                            f"Checking CONNECTED {device_type.upper()}S: [{conn_str}]. "
                            f"Multiple {device_type}s connected. {room_clause} "
                            f"All {device_type}s are in the eligible state. "
                            f"Cannot determine which one to act on. "
                            f"Calling intent_unclear(incomplete)."
                        )
                        examples.append(build_ex(prompt,
                            [{"name": "intent_unclear", "args": {"reason": "incomplete"}}],
                            clarify, avail_r, avail_d, state,
                            user_room=user_room, action_log=action_log,
                            think_trace=think, category="exhaustive_gadget_rules"))

    return examples


def gen_exhaustive_compound_pairs(synonyms_per_combo: int = 20) -> list:
    """
    HYBRID EXHAUSTIVE — Two-action compound commands.
    Logic axes (16 combinations): action1_type × action2_type × satisfaction states
    Total: 16 × 20 = 320 examples.
    """
    examples = []
    SCENE_PROMPTS = {
        "movie_night": "movie night",
        "bedtime":     "bedtime",
        "morning":     "morning mode",
        "away":        "away mode",
        "party":       "party mode",
    }

    for a1_type, a2_type, a1_sat, a2_sat in itertools.product(
        ["light", "door"], ["scene", "thermostat"],
        ["needed", "already_done"], ["needed", "already_done"]
    ):
        for _ in range(synonyms_per_combo):
            avail_r, avail_d = sample_topology(min_rooms=3)
            state = generate_random_state(avail_r, avail_d)
            calls = []
            think_parts = ["Compound request. Evaluating sub-actions."]
            resp_parts  = []

            if a1_type == "light":
                r    = random.choice(avail_r)
                alias = random.choice(ROOM_ALIASES[r])
                req  = random.choice(["on", "off"])
                opp  = "off" if req == "on" else "on"
                if a1_sat == "already_done":
                    apply_force(state, {"lights": {r: req}}, avail_r, avail_d)
                    think_parts.append(
                        f"Sub-action 1: {r} light is already {req}. No tool call needed.")
                    resp_parts.append(f"The {alias} light is already {req}")
                    a1_prompt_frag = f"turn {req} the {alias} light"
                else:
                    apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)
                    calls.append({"name": "toggle_lights", "args": {"room": r, "state": req}})
                    think_parts.append(
                        f"Sub-action 1: toggle_lights(room={r}, state={req}). Needed.")
                    resp_parts.append(f"{alias.title()} light {req}")
                    a1_prompt_frag = f"turn {req} the {alias} light"
            else:
                d    = random.choice(avail_d)
                da   = random.choice(DOOR_ALIASES[d])
                ds   = random.choice(["lock", "unlock"])
                opp_d = "unlock" if ds == "lock" else "lock"
                aw   = "locked" if ds == "lock" else "unlocked"
                opp_aw = "unlocked" if ds == "lock" else "locked"
                if a1_sat == "already_done":
                    apply_force(state, {"doors": {d: aw}}, avail_r, avail_d)
                    think_parts.append(
                        f"Sub-action 1: {d} door is already {aw}. No tool call needed.")
                    resp_parts.append(f"The {da} is already {aw}")
                    a1_prompt_frag = f"{ds} the {da}"
                else:
                    apply_force(state, {"doors": {d: opp_aw}}, avail_r, avail_d)
                    calls.append({"name": "lock_door", "args": {"door": d, "state": ds}})
                    think_parts.append(
                        f"Sub-action 1: lock_door(door={d}, state={ds}). Needed.")
                    resp_parts.append(f"{da.title()} {aw}")
                    a1_prompt_frag = f"{ds} the {da}"

            if a2_type == "scene":
                scene = random.choice(SCENES)
                sc_tr = random.choice(list(SCENE_PROMPTS.values()))
                if a2_sat == "already_done":
                    state["active_scene"] = scene
                    think_parts.append(
                        f"Sub-action 2: scene {scene} is already active. No tool call needed.")
                    resp_parts.append(
                        f"The {scene.replace('_',' ').title()} scene is already active")
                    a2_prompt_frag = f"set {sc_tr} mode"
                else:
                    state["active_scene"] = None
                    calls.append({"name": "set_scene", "args": {"scene": scene}})
                    think_parts.append(
                        f"Sub-action 2: set_scene(scene={scene}). Needed.")
                    resp_parts.append(SCENE_RESP[scene].rstrip("."))
                    a2_prompt_frag = f"set {sc_tr} mode"
            else:
                val = random.randint(MIN_T, MAX_T)
                cur = state["thermostat"]["temperature"]
                mode = "cool" if val < cur else "heat"
                if a2_sat == "already_done":
                    apply_force(state,
                        {"thermostat": {"temperature": val, "mode": mode}},
                        avail_r, avail_d)
                    think_parts.append(
                        f"Sub-action 2: thermostat already at {val}F. No tool call needed.")
                    resp_parts.append(f"Thermostat is already set to {val}°F")
                    a2_prompt_frag = f"set the temperature to {val}"
                else:
                    while val == state["thermostat"]["temperature"]:
                        val = random.randint(MIN_T, MAX_T)
                    mode = "cool" if val < state["thermostat"]["temperature"] else "heat"
                    calls.append({"name": "set_thermostat",
                                  "args": {"temperature": val, "mode": mode}})
                    think_parts.append(
                        f"Sub-action 2: set_thermostat(temperature={val}, mode={mode}). Needed.")
                    resp_parts.append(f"Thermostat set to {val}°F in {mode} mode")
                    a2_prompt_frag = f"set the temperature to {val}"

            prompt = random.choice([
                f"{a1_prompt_frag.capitalize()} and {a2_prompt_frag}.",
                f"Can you {a1_prompt_frag} and also {a2_prompt_frag}?",
                f"Please {a2_prompt_frag} and {a1_prompt_frag}.",
            ])
            think = " ".join(think_parts)
            resp  = " and ".join(resp_parts) + "."
            resp  = resp[0].upper() + resp[1:]
            action_log = build_distractor_log(avail_r, avail_d, n=1) \
                if random.random() < 0.3 else ""
            examples.append(build_ex(
                prompt, calls, resp, avail_r, avail_d, state,
                action_log=action_log,
                think_trace=think, category="exhaustive_compound_pairs"))

    return examples


# ══════════════════════════════════════════════════════════════════════
# PRESERVED + FIXED GENERATORS
# ══════════════════════════════════════════════════════════════════════

def gen_already_satisfied(target: int = 1_800) -> list:
    examples = []
    on_v  = ["Turn on", "Switch on", "Can you turn on", "On"]
    off_v = ["Turn off", "Switch off", "Kill", "Off"]
    lk_v  = ["Lock", "Close", "Secure"]
    ul_v  = ["Unlock", "Open", "Open up"]
 
    for _ in range(target):
        choice = random.choice(["light_on", "light_off", "door_lk", "door_ul"])
 
        if choice == "light_on":
            r = random.choice(ALL_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[r])
            state = generate_random_state(avail_r, avail_d)
            alias = random.choice(ROOM_ALIASES[r])
            apply_force(state, {"lights": {r: "on"}}, avail_r, avail_d)
            noise_room = random.choice(["", r, random.choice(avail_r)])
            loc_str = (f"current_user_room='{noise_room}'"
                       if noise_room else "user room is unknown")
            # FIX-1: open with STATE check so the model reads state before deciding
            think = (
                f"Checking STATE: {r}=on. "
                f"User explicitly named the '{alias}' light . "
                f"User wants on. "
                f"Match — STATE already shows on. No tool call needed."
            )
            examples.append(build_ex(
                f"{random.choice(on_v)} the {alias} light.", [],
                f"The {alias} light is already on.",
                avail_r, avail_d, state,
                user_room=noise_room,
                think_trace=think, category="already_satisfied"))
 
        elif choice == "light_off":
            r = random.choice(ALL_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[r])
            state = generate_random_state(avail_r, avail_d)
            alias = random.choice(ROOM_ALIASES[r])
            apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
            noise_room = random.choice(["", r, random.choice(avail_r)])
            loc_str = (f"current_user_room='{noise_room}'"
                       if noise_room else "user room is unknown")
            think = (
                f"Checking STATE: {r}=off. "
                f"User explicitly named the '{alias}' light . "
                f"User wants off. "
                f"Match — STATE already shows off. No tool call needed."
            )
            examples.append(build_ex(
                f"{random.choice(off_v)} the {alias} light.", [],
                f"The {alias} light is already off.",
                avail_r, avail_d, state,
                user_room=noise_room,
                think_trace=think, category="already_satisfied"))
 
        elif choice == "door_lk":
            d = random.choice(ALL_DOORS)
            avail_r, avail_d = sample_topology(required_doors=[d])
            state = generate_random_state(avail_r, avail_d)
            alias = random.choice(DOOR_ALIASES[d])
            apply_force(state, {"doors": {d: "locked"}}, avail_r, avail_d)
            noise_room = random.choice(["", random.choice(avail_r)])
            loc_str = (f"current_user_room='{noise_room}'"
                       if noise_room else "user room is unknown")
            think = (
                f"Checking STATE: {d}=locked. "
                f"User explicitly named the '{alias}' . "
                f"User wants locked. "
                f"Match — STATE already shows locked. No tool call needed."
            )
            examples.append(build_ex(
                f"{random.choice(lk_v)} the {alias}.", [],
                f"The {alias} is already locked.",
                avail_r, avail_d, state,
                user_room=noise_room,
                think_trace=think, category="already_satisfied"))
 
        else:  # door_ul
            d = random.choice(ALL_DOORS)
            avail_r, avail_d = sample_topology(required_doors=[d])
            state = generate_random_state(avail_r, avail_d)
            alias = random.choice(DOOR_ALIASES[d])
            apply_force(state, {"doors": {d: "unlocked"}}, avail_r, avail_d)
            noise_room = random.choice(["", random.choice(avail_r)])
            loc_str = (f"current_user_room='{noise_room}'"
                       if noise_room else "user room is unknown")
            think = (
                f"Checking STATE: {d}=unlocked. "
                f"User explicitly named the '{alias}' . "
                f"User wants unlocked. "
                f"Match — STATE already shows unlocked. No tool call needed."
            )
            examples.append(build_ex(
                f"{random.choice(ul_v)} the {alias}.", [],
                f"The {alias} is already unlocked.",
                avail_r, avail_d, state,
                user_room=noise_room,
                think_trace=think, category="already_satisfied"))
 
    return examples

def gen_action_required(target: int = 3_000) -> list:
    examples = []
    on_t  = ["Turn on the {r} light.", "Switch on the {r} light.",
              "Light up the {r}.", "Can you turn on the {r} light?",
              "On the {r} light."]
    off_t = ["Turn off the {r} light.", "Switch off the {r} light.",
              "Kill the {r} light.", "Lights off in the {r}.",
              "Off the {r} light."]
    lk_t  = ["Lock the {d}.", "Close the {d}.", "Secure the {d}."]
    ul_t  = ["Unlock the {d}.", "Open the {d}.", "Open up the {d}."]
 
    for _ in range(target):
        if random.random() < 0.5:
            # ── light branch ────────────────────────────────────────────
            r = random.choice(ALL_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[r])
            state = generate_random_state(avail_r, avail_d)
            alias = random.choice(ROOM_ALIASES[r])
            distractor = (build_distractor_log(avail_r, avail_d, n=1)
                          if random.random() < 0.5 else "")
            noise_room = random.choice(["", r, random.choice(avail_r)])
            loc_str = (f"current_user_room='{noise_room}'"
                       if noise_room else "user room is unknown")
 
            if random.random() < 0.5:
                # want ON → current must be OFF
                apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
                current_state = "off"
                desired       = "on"
                think = (
                    f"Checking STATE: {r}={current_state}. "
                    f"User explicitly named the '{alias}' light. "
                    f"Mismatch — STATE shows {current_state}, user wants {desired}. "
                    f"Calling toggle_lights(room={r}, state={desired})."
                )
                examples.append(build_ex(
                    random.choice(on_t).format(r=alias),
                    [{"name": "toggle_lights", "args": {"room": r, "state": "on"}}],
                    f"The {alias} light is now on.",
                    avail_r, avail_d, state,
                    user_room=noise_room, action_log=distractor,
                    think_trace=think, category="action_required"))
            else:
                # want OFF → current must be ON
                apply_force(state, {"lights": {r: "on"}}, avail_r, avail_d)
                current_state = "on"
                desired       = "off"
                think = (
                    f"Checking STATE: {r}={current_state}. "
                    f"User explicitly named the '{alias}' light. "
                    f"Mismatch — STATE shows {current_state}, user wants {desired}. "
                    f"Calling toggle_lights(room={r}, state={desired})."
                )
                examples.append(build_ex(
                    random.choice(off_t).format(r=alias),
                    [{"name": "toggle_lights", "args": {"room": r, "state": "off"}}],
                    f"The {alias} light is now off.",
                    avail_r, avail_d, state,
                    user_room=noise_room, action_log=distractor,
                    think_trace=think, category="action_required"))
 
        else:
            # ── door branch ─────────────────────────────────────────────
            d = random.choice(ALL_DOORS)
            avail_r, avail_d = sample_topology(required_doors=[d])
            state = generate_random_state(avail_r, avail_d)
            alias = random.choice(DOOR_ALIASES[d])
            distractor = (build_distractor_log(avail_r, avail_d, n=1)
                          if random.random() < 0.5 else "")
            noise_room = random.choice(["", random.choice(avail_r)])
            loc_str = (f"current_user_room='{noise_room}'"
                       if noise_room else "user room is unknown")
 
            if random.random() < 0.5:
                # want LOCK → current must be unlocked
                apply_force(state, {"doors": {d: "unlocked"}}, avail_r, avail_d)
                current_state = "unlocked"
                desired_state = "locked"
                think = (
                    f"Checking STATE: {d}={current_state}. "
                    f"User explicitly named the '{alias}'. "
                    f"User wants {desired_state}. "
                    f"Mismatch — STATE shows {current_state}, user wants {desired_state}. "
                    f"Calling lock_door(door={d}, state=lock)."
                )
                examples.append(build_ex(
                    random.choice(lk_t).format(d=alias),
                    [{"name": "lock_door", "args": {"door": d, "state": "lock"}}],
                    f"The {alias} is now locked.",
                    avail_r, avail_d, state,
                    user_room=noise_room, action_log=distractor,
                    think_trace=think, category="action_required"))
            else:
                # want UNLOCK → current must be locked
                apply_force(state, {"doors": {d: "locked"}}, avail_r, avail_d)
                current_state = "locked"
                desired_state = "unlocked"
                think = (
                    f"Checking STATE: {d}={current_state}. "
                    f"User explicitly named the '{alias}'. "
                    f"User wants {desired_state}. "
                    f"Mismatch — STATE shows {current_state}, user wants {desired_state}. "
                    f"Calling lock_door(door={d}, state=unlock)."
                )
                examples.append(build_ex(
                    random.choice(ul_t).format(d=alias),
                    [{"name": "lock_door", "args": {"door": d, "state": "unlock"}}],
                    f"The {alias} is now unlocked.",
                    avail_r, avail_d, state,
                    user_room=noise_room, action_log=distractor,
                    think_trace=think, category="action_required"))
 
    return examples


def gen_user_room_lights(target: int = 2_500) -> list:
    """GAP A reinforcement: 80% of examples have cross-room distractor."""
    examples = []
    on_p  = [
        "Turn the light on.", "Lights on.", "Light on please.", "It's dark in here.",
        "I can't see.", "Switch on the light.", "On the light.", "On this light.",
        "Turn the lights on.", "Lights on please.", "All lights on in here.",
    ]
    off_p = [
        "Turn the light off.", "Lights off.", "Light off please.", "Too bright in here.",
        "Switch off the light.", "Cut the light.", "Off the light.", "Off this light.",
        "Off the lights.", "Lights off please.", "Kill the lights in here.", "Turn the lights off.",
    ]
    for _ in range(target):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        alias = random.choice(ROOM_ALIASES[r])
        action_log = ""
        if random.random() < 0.80 and len(avail_r) > 1:
            dist_mins  = random.randint(5, 15)
            action_log = get_cross_room_txn(avail_r, avail_d, r, dist_mins)
            if random.random() < 0.4:
                more = build_distractor_log(avail_r, avail_d, n=1,
                                            start_mins=dist_mins + random.randint(8, 15))
                action_log = action_log + "\n" + more
        s   = random.choice(["on", "off"])
        opp = "off" if s == "on" else "on"
        apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)
        prompt = random.choice(on_p if s == "on" else off_p)
        
        think = (
            f"User is in '{r}'. User said '{prompt}'. "
            f"Generic 'the light' + current_user_room='{r}' → {r} light. "
            f"Current state is '{opp}'. Target state is '{s}'. "
            f"Calling toggle_lights(room={r}, state={s})."
        )

        examples.append(build_ex(prompt,
            [{"name": "toggle_lights", "args": {"room": r, "state": s}}],
            f"The {alias} light is now {s}.",
            avail_r, avail_d, state,
            user_room=r, action_log=action_log,
            think_trace=think, category="user_room_lights"))
    return examples


def gen_user_room_doors(target: int = 1_200) -> list:
    examples = []
    ul_p = ["Open this door.", "Unlock this door.", "Open the door.",
            "Let me in.", "Door open please."]
    lk_p = ["Close this door.", "Lock this door.", "Lock the door.",
            "Shut it.", "Secure the door."]
    for _ in range(target):
        d = random.choice(["bedroom", "bathroom", "office", "kitchen", "living_room"])
        avail_r, avail_d = sample_topology(required_doors=[d])
        state = generate_random_state(avail_r, avail_d)
        alias = random.choice(DOOR_ALIASES[d])
        action_log = ""
        if random.random() < 0.7 and len(avail_d) > 1:
            other_d   = random.choice([x for x in avail_d if x != d])
            dist_mins = random.randint(5, 18)
            action_log = fmt_txn(dist_mins,
                                 [f"lock_door(door={other_d}, state=lock)"],
                                 f"{DOOR_DISPLAY[other_d]} locked.")
        s   = random.choice(["lock", "unlock"])
        opp = "unlock" if s == "lock" else "lock"
        aw  = "closed" if s == "lock" else "open"
        apply_force(state, {"doors": {d: opp}}, avail_r, avail_d)
        prompt = random.choice(lk_p if s == "lock" else ul_p)
        think  = (
            f"User is in '{d}' area. Said '{prompt}'. "
            f"current_user_room resolves to '{d}'. "
            f"Calling lock_door(door={d}, state={s})."
        )
        examples.append(build_ex(prompt,
            [{"name": "lock_door", "args": {"door": d, "state": s}}],
            f"The {alias} is now {aw}.", avail_r, avail_d, state,
            user_room=d, action_log=action_log,
            think_trace=think, category="user_room_doors"))
    return examples


def gen_bulk_state_aware(target: int = 3_500) -> list:
    examples = []
    off_on   = ["Turn off what's on.",
                 "Kill the {lw} that are on.",
                 "Off the {lw} that are on.",
                 "Turn off all {lw} currently on."]
    all_on   = ["Turn on all the {lw}.", "All {lw} on.", "Every light on.",
                 "Lights on everywhere.", "On all the {lw}."]
    all_off  = ["Turn off all the {lw}.", "All {lw} off.", "Kill all the {lw}.",
                 "Every light off.", "Off all the {lw}."]
    lock_p   = ["Lock all the {dw}.", "Lock everything.", "Secure all {dw}."]
    unlock_p = ["Unlock all the {dw}.", "Open all the {dw}.", "Unlock everything."]
    lock_unl = [
        "Lock the {dw} that are open.", "Close all open {dw}.",
        "Lock what's unlocked.",        "Secure any open {dw}.",
        "Close any {dw} that is open.", "Lock the {dw} that is unlocked.",
        "Close the unlocked {dw}.",     "Secure the open {dw}.",
        "Lock any {dw} that isn't locked.", "Close every open {dw}.",
    ]
    for _ in range(int(target * 1.1)):
        avail_r, avail_d = sample_topology()
        state     = generate_random_state(avail_r, avail_d)
        combo     = random.choice(["off_on", "all_on", "all_off", "lock",
                                   "unlock", "lock_unl"])
        user_room = random.choice(["", random.choice(avail_r)])
        
        lw = typo_word("lights", LIGHT_TYPOS)
        dw = typo_word("doors",  DOOR_TYPOS)
 
        if combo == "off_on":
            on_r = [r for r in avail_r if state["lights"][r]["state"] == "on"]
            if not on_r:
                summary = ", ".join(
                    f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
                prompt_chosen = random.choice(off_on).format(lw=lw)
                think = (
                    f"User said '{prompt_chosen}'. Single-device bulk command — lights only. "
                    f"Global light scope. Checking ALL connected lights: {summary}. "
                    f"Result: 0 lights currently on — ALL already off. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt_chosen, [],
                    "All lights are already off.",
                    avail_r, avail_d, state,
                    user_room=user_room, think_trace=think,
                    category="bulk_state_aware"))
                continue
            summary  = ", ".join(
                f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
            on_names = ", ".join(ROOM_DISPLAY[r] for r in on_r)
            prompt_chosen = random.choice(off_on).format(lw=lw)
            think = (
                f"User said '{prompt_chosen}'. Single-device bulk command — lights only. "
                f"Global light scope. Checking ALL connected lights: {summary}. "
                f"{len(on_r)} light(s) currently on ({on_names}). "
                f"Issuing {len(on_r)} toggle_lights(state=off) calls individually."
            )
            calls = [{"name": "toggle_lights", "args": {"room": r, "state": "off"}}
                     for r in on_r]
            resp  = " ".join(f"{ROOM_DISPLAY[r].title()} light off." for r in on_r)
            examples.append(build_ex(prompt_chosen, calls, resp,
                avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="bulk_state_aware"))
 
        elif combo == "lock_unl":
            unl_d = [d for d in avail_d if state["doors"][d] == "unlocked"]
            if not unl_d:
                summary = ", ".join(
                    f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
                prompt_chosen = random.choice(lock_unl).format(dw=dw)
                think = (
                    f"User said '{prompt_chosen}'. Single-device bulk command — doors only. "
                    f"Global door scope. Checking ALL connected doors: {summary}. "
                    f"Result: 0 doors currently unlocked — ALL already locked. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt_chosen, [],
                    "All doors are already locked.",
                    avail_r, avail_d, state,
                    user_room=user_room, think_trace=think,
                    category="bulk_state_aware"))
                continue
            summary   = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            unl_names = ", ".join(DOOR_DISPLAY[d] for d in unl_d)
            prompt_chosen = random.choice(lock_unl).format(dw=dw)
            think = (
                f"User said '{prompt_chosen}'. Single-device bulk command — doors only. "
                f"Global door scope. Checking ALL connected doors: {summary}. "
                f"{len(unl_d)} door(s) currently unlocked ({unl_names}). "
                f"Issuing {len(unl_d)} lock_door(state=lock) calls individually."
            )
            calls = [{"name": "lock_door", "args": {"door": d, "state": "lock"}}
                     for d in unl_d]
            resp  = " ".join(f"{DOOR_DISPLAY[d].title()} locked." for d in unl_d)
            examples.append(build_ex(prompt_chosen, calls, resp,
                avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="bulk_state_aware"))
 
        elif combo == "all_on":
            off_r = [r for r in avail_r if state["lights"][r]["state"] == "off"]
            if not off_r:
                summary = ", ".join(
                    f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
                prompt_chosen = random.choice(all_on).format(lw=lw)
                think = (
                    f"User said '{prompt_chosen}'. Single-device bulk command — lights only. "
                    f"Global light scope. Checking ALL connected lights: {summary}. "
                    f"Result: 0 lights currently off — ALL already on. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt_chosen, [],
                    "All lights are already on.",
                    avail_r, avail_d, state,
                    user_room=user_room, think_trace=think,
                    category="bulk_state_aware"))
                continue
            summary   = ", ".join(f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
            off_names = ", ".join(ROOM_DISPLAY[r] for r in off_r)
            prompt_chosen = random.choice(all_on).format(lw=lw)
            think = (
                f"User said '{prompt_chosen}'. Single-device bulk command — lights only. "
                f"Global light scope. Checking ALL connected lights: {summary}. "
                f"{len(off_r)} light(s) currently off ({off_names}). "
                f"Issuing {len(off_r)} toggle_lights(state=on) calls individually."
            )
            calls = [{"name": "toggle_lights", "args": {"room": r, "state": "on"}} for r in off_r]
            resp  = " ".join(f"{ROOM_DISPLAY[r].title()} light on." for r in off_r)
            examples.append(build_ex(prompt_chosen, calls, resp, avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="bulk_state_aware"))
 
        elif combo == "all_off":
            on_r = [r for r in avail_r if state["lights"][r]["state"] == "on"]
            if not on_r:
                summary = ", ".join(
                    f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
                prompt_chosen = random.choice(all_off).format(lw=lw)
                think = (
                    f"User said '{prompt_chosen}'. Single-device bulk command — lights only. "
                    f"Global light scope. Checking ALL connected lights: {summary}. "
                    f"Result: 0 lights currently on — ALL already off. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt_chosen, [],
                    "All lights are already off.",
                    avail_r, avail_d, state,
                    user_room=user_room, think_trace=think,
                    category="bulk_state_aware"))
                continue
            summary  = ", ".join(f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
            on_names = ", ".join(ROOM_DISPLAY[r] for r in on_r)
            prompt_chosen = random.choice(all_off).format(lw=lw)
            think = (
                f"User said '{prompt_chosen}'. Single-device bulk command — lights only. "
                f"Global light scope. Checking ALL connected lights: {summary}. "
                f"{len(on_r)} light(s) currently on ({on_names}). "
                f"Issuing {len(on_r)} toggle_lights(state=off) calls individually."
            )
            calls = [{"name": "toggle_lights", "args": {"room": r, "state": "off"}} for r in on_r]
            resp  = " ".join(f"{ROOM_DISPLAY[r].title()} light off." for r in on_r)
            examples.append(build_ex(prompt_chosen, calls, resp, avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="bulk_state_aware"))
 
        elif combo == "lock":
            unlocked = [d for d in avail_d if state["doors"][d] == "unlocked"]
            if not unlocked:
                summary = ", ".join(
                    f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
                prompt_chosen = random.choice(lock_p).format(dw=dw)
                think = (
                    f"User said '{prompt_chosen}'. Single-device bulk command — doors only. "
                    f"Global door scope. Checking ALL connected doors: {summary}. "
                    f"Result: 0 doors currently unlocked — ALL already locked. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt_chosen, [],
                    "All doors are already locked.",
                    avail_r, avail_d, state,
                    user_room=user_room, think_trace=think,
                    category="bulk_state_aware"))
                continue
            summary   = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            unl_names = ", ".join(DOOR_DISPLAY[d] for d in unlocked)
            prompt_chosen = random.choice(lock_p).format(dw=dw)
            think = (
                f"User said '{prompt_chosen}'. Single-device bulk command — doors only. "
                f"Global door scope. Checking ALL connected doors: {summary}. "
                f"{len(unlocked)} door(s) currently unlocked ({unl_names}). "
                f"Issuing {len(unlocked)} lock_door(state=lock) calls individually."
            )
            calls = [{"name": "lock_door", "args": {"door": d, "state": "lock"}} for d in unlocked]
            resp  = " ".join(f"{DOOR_DISPLAY[d].title()} locked." for d in unlocked)
            examples.append(build_ex(prompt_chosen, calls, resp, avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="bulk_state_aware"))
 
        else:  # unlock
            locked = [d for d in avail_d if state["doors"][d] == "locked"]
            if not locked:
                summary = ", ".join(
                    f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
                prompt_chosen = random.choice(unlock_p).format(dw=dw)
                think = (
                    f"User said '{prompt_chosen}'. Single-device bulk command — doors only. "
                    f"Global door scope. Checking ALL connected doors: {summary}. "
                    f"Result: 0 doors currently locked — ALL already unlocked. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt_chosen, [],
                    "All doors are already unlocked.",
                    avail_r, avail_d, state,
                    user_room=user_room, think_trace=think,
                    category="bulk_state_aware"))
                continue
            summary  = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            lk_names = ", ".join(DOOR_DISPLAY[d] for d in locked)
            prompt_chosen = random.choice(unlock_p).format(dw=dw)
            think = (
                f"User said '{prompt_chosen}'. Single-device bulk command — doors only. "
                f"Global door scope. Checking ALL connected doors: {summary}. "
                f"{len(locked)} door(s) currently locked ({lk_names}). "
                f"Issuing {len(locked)} lock_door(state=unlock) calls individually."
            )
            calls = [{"name": "lock_door", "args": {"door": d, "state": "unlock"}} for d in locked]
            resp  = " ".join(f"{DOOR_DISPLAY[d].title()} unlocked." for d in locked)
            examples.append(build_ex(prompt_chosen, calls, resp, avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="bulk_state_aware"))
 
    random.shuffle(examples)
    return examples[:target]


def gen_state_report_queries(target: int = 3_500) -> list:
    """
    Trains the model to scan the full [STATE:] block and compose a
    human-readable status reply without calling any tool.
 
    Think-trace pattern (consistent with gen_state_grounding_stress):
        "User asks '...'. Reading STATE: <exact state values>.
         <filtered result>. Composing text reply. No tool call needed."
    """
    examples = []
 
    # ── natural-language helpers ──────────────────────────────────────────
    def eng_list(names):
        """['a','b','c'] → 'a, b, and c'"""
        if not names:   return "none"
        if len(names) == 1: return names[0]
        if len(names) == 2: return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"
 
    def room_names(rooms):
        return eng_list([ROOM_DISPLAY[r] for r in rooms])
 
    def door_names(doors):
        return eng_list([DOOR_DISPLAY[d] for d in doors])
 
    # ── query banks ───────────────────────────────────────────────────────
    LIGHTS_ON_Q = [
        "Which lights are on?", "What lights are on?",
        "Are any lights on?", "Which rooms have lights on?",
        "Tell me which lights are on.", "Any lights on right now?",
        "What lights are currently on?", "Which rooms are lit up?",
        "What's on in the house?", "Are any lights still on?",
    ]
    LIGHTS_OFF_Q = [
        "Which lights are off?", "Any lights off?",
        "Which rooms have lights off?", "What lights aren't on?",
        "Tell me which lights are off.", "Which lights are currently off?",
    ]
    LIGHTS_FULL_Q = [
        "What's the light status?", "Give me a lights rundown.",
        "How are the lights?", "Lights status please.",
        "Which lights are on and which are off?",
        "What are the lights doing?", "Lights status?",
        "Light situation?", "What's the situation with the lights?",
    ]
    DOORS_LOCKED_Q = [
        "Which doors are locked?", "Are all the doors locked?",
        "What doors are locked?", "Is everything locked?",
        "Which doors are secure?", "Tell me which doors are locked.",
        "Are the doors locked?", "Everything locked up?",
    ]
    DOORS_UNLOCKED_Q = [
        "Which doors are unlocked?", "Any doors open?",
        "What doors are unlocked?", "Which doors are open?",
        "Any unlocked doors?", "Are any doors unlocked?",
        "Which doors aren't locked?",
    ]
    DOORS_FULL_Q = [
        "Door status?", "What's the door status?",
        "How are the doors?", "Give me a door rundown.",
        "Which doors are locked and which are open?",
        "What's the status of the doors?", "Doors status please.",
        "What are the doors doing?",
    ]
    THERM_Q = [
        "What's the temperature set to?", "What's the thermostat at?",
        "What temp is it set to?", "Thermostat status?",
        "What's the current thermostat setting?",
        "What temperature is the house set to?",
        "Is the heating or cooling on?", "What mode is the thermostat in?",
        "What's the temp?", "Thermostat reading?",
        "How warm is the house set?", "What's the house temperature?",
    ]
    SCENE_Q = [
        "What scene is active?", "Is any scene on?",
        "What mode is the house in?", "Any scenes active?",
        "Which scene is set?", "What's the current scene?",
        "Is there a scene running?", "What home mode is on?",
    ]
    TV_Q = [
        "Is the TV on?", "What's the TV doing?", "TV status?",
        "Is the TV running?", "Which TVs are on?",
        "Are any TVs on?", "TV on or off?", "What are the TVs doing?",
    ]
    FAN_Q = [
        "Is the fan on?", "Fan status?", "Are any fans running?",
        "Which fans are on?", "What are the fans doing?",
        "Are the fans on?", "Fan running?", "Which fans are running?",
    ]
    SPEAKER_Q = [
        "What's playing?", "Is music playing?", "Speaker status?",
        "What's the speaker doing?", "Is anything playing?",
        "What's on the speaker?", "Music status?",
        "Is the music on?", "What music is playing?",
        "Is the speaker running?",
    ]
    MULTI_LIGHTS_TEMP_Q = [
        "Which lights are on and what's the temp?",
        "Lights and thermostat status?",
        "What lights are on and what temperature is it set to?",
        "Tell me about the lights and the thermostat.",
        "What's the light situation and the temp?",
        "Lights on and thermostat setting?",
    ]
    MULTI_LIGHTS_DOORS_Q = [
        "Which lights are on and are the doors locked?",
        "Lights and door status?",
        "What lights are on and which doors are unlocked?",
        "Give me the lights and doors status.",
        "What's on and what's unlocked?",
        "Lights status and door status?",
        "Which lights are on and which doors are open?",
        "Lights and doors?",
    ]
    MULTI_LIGHTS_TV_Q = [
        "Which lights are on and is the TV on?",
        "Lights and TV status?",
        "What's on — lights and TV?",
        "Are any lights on and is the TV running?",
        "Lights and TV?",
    ]
    MULTI_DOORS_SCENE_Q = [
        "Are the doors locked and what scene is on?",
        "Door status and current scene?",
        "Which doors are locked and what mode is active?",
        "Scene and door status?",
    ]
    MULTI_TEMP_SCENE_Q = [
        "What's the temp and what scene is active?",
        "Thermostat and scene status?",
        "What temperature and what mode?",
        "Current scene and thermostat?",
    ]
    MULTI_TV_SPEAKER_Q = [
        "Is the TV on and what's playing?",
        "TV and speaker status?",
        "What's the TV and music doing?",
        "Are the TV and speaker on?",
    ]
    FULL_HOME_Q = [
        "What's the status of everything?",
        "Full home status.",
        "Give me a home report.",
        "What's going on in the house?",
        "Home status?",
        "Status report.",
        "What's everything doing right now?",
        "Tell me the status of the whole house.",
        "What's running in the house?",
        "Give me a full rundown.",
        "How is everything?",
        "What's the house doing?",
        "Full status report please.",
        "Everything status?",
        "What's on and what's off in the house?",
        "Can you give me a complete home status?",
        "Run me through everything.",
        "What's the state of the house?",
        "Full report.",
        "Give me the whole picture.",
    ]
 
    CATS    = [
        "lights_on", "lights_off", "lights_full",
        "doors_locked", "doors_unlocked", "doors_full",
        "thermostat", "scene", "tv", "fan", "speaker",
        "multi_lights_temp", "multi_lights_doors", "multi_lights_tv",
        "multi_doors_scene", "multi_temp_scene", "multi_tv_speaker",
        "full_home",
    ]
    WEIGHTS = [
        6, 3, 5,        # lights
        5, 3, 4,        # doors
        4, 3, 3, 3, 3,  # single gadgets
        5, 5, 4,        # multi with lights
        3, 3, 3,        # other multi
        18,             # full home (highest — most training value)
    ]
 
    attempts = 0
    while len(examples) < target and attempts < target * 5:
        attempts += 1
        avail_r, avail_d = sample_topology()
        state     = generate_random_state(avail_r, avail_d)
        cat       = random.choices(CATS, weights=WEIGHTS)[0]
        act_log   = (build_distractor_log(avail_r, avail_d, n=1)
                     if random.random() < 0.35 else "")
        user_room = random.choice([""] + avail_r)
 
        # ── pre-compute filtered results for every device type ─────────
        on_r  = [r for r in avail_r if state["lights"][r]["state"] == "on"]
        off_r = [r for r in avail_r if state["lights"][r]["state"] == "off"]
        locked   = [d for d in avail_d if state["doors"][d] == "locked"]
        unlocked = [d for d in avail_d if state["doors"][d] == "unlocked"]
        t     = state["thermostat"]["temperature"]
        m     = state["thermostat"]["mode"]
        scene = state.get("active_scene")
        tv_d  = state.get("tv",      {})
        sp_d  = state.get("speaker", {})
        fan_d = state.get("fan",     {})
 
        # ── state summary strings for think traces ──────────────────────
        l_ss = ", ".join(
            f"{r}:{state['lights'][r]['state']}" for r in sorted(avail_r))
        d_ss = ", ".join(
            f"{d}:{state['doors'][d]}" for d in sorted(avail_d))
        tv_ss  = (", ".join(f"{r}:{tv_d[r]}"  for r in sorted(tv_d))
                  if tv_d else "none")
        sp_ss  = (", ".join(f"{r}:{sp_d[r]}"  for r in sorted(sp_d))
                  if sp_d else "none")
        fan_ss = (", ".join(
            f"{r}:{fan_d[r]['state']}({fan_d[r]['speed']})"
            for r in sorted(fan_d)) if fan_d else "none")
 
        resp  = ""
        think = ""
        prompt = ""
 
        # ── per-category logic ──────────────────────────────────────────
 
        if cat == "lights_on":
            prompt = random.choice(LIGHTS_ON_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: lights={{{l_ss}}}. "
                f"Lights currently on: {on_r if on_r else 'none'}. "
                f"Composing text reply. No tool call needed."
            )
            if not on_r:
                resp = "All lights are currently off."
            elif len(on_r) == len(avail_r):
                resp = "All lights are on."
            else:
                resp = (
                    f"The {room_names(on_r)} "
                    f"{'light is' if len(on_r) == 1 else 'lights are'} on. "
                    f"The {room_names(off_r)} "
                    f"{'light is' if len(off_r) == 1 else 'lights are'} off."
                )
 
        elif cat == "lights_off":
            prompt = random.choice(LIGHTS_OFF_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: lights={{{l_ss}}}. "
                f"Lights currently off: {off_r if off_r else 'none'}. "
                f"Composing text reply. No tool call needed."
            )
            if not off_r:
                resp = "All lights are currently on."
            elif len(off_r) == len(avail_r):
                resp = "All lights are off."
            else:
                resp = (
                    f"The {room_names(off_r)} "
                    f"{'light is' if len(off_r) == 1 else 'lights are'} off. "
                    f"The {room_names(on_r)} "
                    f"{'light is' if len(on_r) == 1 else 'lights are'} on."
                )
 
        elif cat == "lights_full":
            prompt = random.choice(LIGHTS_FULL_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: lights={{{l_ss}}}. "
                f"On: {on_r}. Off: {off_r}. "
                f"Composing full lights report. No tool call needed."
            )
            if not on_r:
                resp = "All lights are off."
            elif len(on_r) == len(avail_r):
                resp = "All lights are on."
            else:
                resp = (
                    f"On: {room_names(on_r)}. "
                    f"Off: {room_names(off_r)}."
                )
 
        elif cat == "doors_locked":
            prompt = random.choice(DOORS_LOCKED_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: doors={{{d_ss}}}. "
                f"Locked: {locked}. Unlocked: {unlocked}. "
                f"Composing text reply. No tool call needed."
            )
            if len(locked) == len(avail_d):
                resp = "All doors are locked."
            elif not locked:
                resp = "No doors are locked — all are unlocked."
            else:
                resp = (
                    f"Locked: {door_names(locked)}. "
                    f"Unlocked: {door_names(unlocked)}."
                )
 
        elif cat == "doors_unlocked":
            prompt = random.choice(DOORS_UNLOCKED_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: doors={{{d_ss}}}. "
                f"Unlocked: {unlocked}. Locked: {locked}. "
                f"Composing text reply. No tool call needed."
            )
            if not unlocked:
                resp = "All doors are locked — none are unlocked."
            elif len(unlocked) == len(avail_d):
                resp = "All doors are unlocked."
            else:
                resp = (
                    f"The {door_names(unlocked)} "
                    f"{'is' if len(unlocked) == 1 else 'are'} unlocked. "
                    f"The {door_names(locked)} "
                    f"{'is' if len(locked) == 1 else 'are'} locked."
                )
 
        elif cat == "doors_full":
            prompt = random.choice(DOORS_FULL_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: doors={{{d_ss}}}. "
                f"Locked: {locked}. Unlocked: {unlocked}. "
                f"Composing full door report. No tool call needed."
            )
            if len(locked) == len(avail_d):
                resp = "All doors are locked."
            elif not locked:
                resp = "All doors are unlocked."
            else:
                resp = (
                    f"Locked: {door_names(locked)}. "
                    f"Unlocked: {door_names(unlocked)}."
                )
 
        elif cat == "thermostat":
            prompt = random.choice(THERM_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: thermostat={t}F/{m}. "
                f"Composing text reply. No tool call needed."
            )
            resp = f"The thermostat is set to {t}°F in {m} mode."
 
        elif cat == "scene":
            prompt = random.choice(SCENE_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: active_scene={scene or 'none'}. "
                f"Composing text reply. No tool call needed."
            )
            resp = (
                f"The {scene.replace('_', ' ').title()} scene is active."
                if scene else
                "No scene is currently active."
            )
 
        elif cat == "tv":
            if not tv_d: continue
            on_tvs  = [r for r in tv_d if tv_d[r] == "on"]
            off_tvs = [r for r in tv_d if tv_d[r] == "off"]
            prompt = random.choice(TV_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: tv={{{tv_ss}}}. "
                f"On: {on_tvs}. Off: {off_tvs}. "
                f"Composing text reply. No tool call needed."
            )
            if not on_tvs:
                resp = "No TVs are currently on."
            elif len(on_tvs) == 1:
                resp = f"The {ROOM_DISPLAY[on_tvs[0]]} TV is on."
                if off_tvs:
                    resp += f" The {room_names(off_tvs)} TV is off."
            else:
                resp = f"TVs on: {room_names(on_tvs)}."
                if off_tvs:
                    resp += f" TVs off: {room_names(off_tvs)}."
 
        elif cat == "fan":
            if not fan_d: continue
            on_fans  = [r for r in fan_d if fan_d[r]["state"] == "on"]
            off_fans = [r for r in fan_d if fan_d[r]["state"] == "off"]
            prompt = random.choice(FAN_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: fan={{{fan_ss}}}. "
                f"On: {on_fans}. Off: {off_fans}. "
                f"Composing text reply. No tool call needed."
            )
            if not on_fans:
                resp = "No fans are running."
            elif len(on_fans) == 1:
                spd  = fan_d[on_fans[0]]["speed"]
                resp = f"The {ROOM_DISPLAY[on_fans[0]]} fan is on at {spd} speed."
                if off_fans:
                    resp += f" The {room_names(off_fans)} fan is off."
            else:
                parts_list = [
                    f"{ROOM_DISPLAY[r]} ({fan_d[r]['speed']})" for r in on_fans
                ]
                resp = "Fans on: " + ", ".join(parts_list) + "."
                if off_fans:
                    resp += f" Fans off: {room_names(off_fans)}."
 
        elif cat == "speaker":
            if not sp_d: continue
            playing = [r for r in sp_d if sp_d[r] == "playing"]
            paused  = [r for r in sp_d if sp_d[r] == "paused"]
            stopped = [r for r in sp_d if sp_d[r] == "stopped"]
            prompt = random.choice(SPEAKER_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: speaker={{{sp_ss}}}. "
                f"Playing: {playing}. Paused: {paused}. Stopped: {stopped}. "
                f"Composing text reply. No tool call needed."
            )
            parts = []
            if playing:
                parts.append(
                    f"{room_names(playing)} "
                    f"{'speaker is' if len(playing) == 1 else 'speakers are'} playing"
                )
            if paused:
                parts.append(
                    f"{room_names(paused)} "
                    f"{'speaker is' if len(paused) == 1 else 'speakers are'} paused"
                )
            if stopped and (playing or paused):
                parts.append(
                    f"{room_names(stopped)} "
                    f"{'speaker is' if len(stopped) == 1 else 'speakers are'} stopped"
                )
            resp = (". ".join(p.capitalize() for p in parts) + "."
                    if parts else "No speakers are playing.")
 
        # ── MULTI: lights + thermostat ──────────────────────────────────
        elif cat == "multi_lights_temp":
            prompt = random.choice(MULTI_LIGHTS_TEMP_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: lights={{{l_ss}}}, thermostat={t}F/{m}. "
                f"Lights on: {on_r}. Lights off: {off_r}. "
                f"Composing lights + thermostat reply. No tool call needed."
            )
            if not on_r:
                l_part = "All lights are off."
            elif len(on_r) == len(avail_r):
                l_part = "All lights are on."
            else:
                l_part = (
                    f"Lights on: {room_names(on_r)}. "
                    f"Lights off: {room_names(off_r)}."
                )
            resp = f"{l_part} Thermostat: {t}°F in {m} mode."
 
        # ── MULTI: lights + doors ───────────────────────────────────────
        elif cat == "multi_lights_doors":
            prompt = random.choice(MULTI_LIGHTS_DOORS_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: lights={{{l_ss}}}, doors={{{d_ss}}}. "
                f"Lights on: {on_r}. Unlocked doors: {unlocked}. "
                f"Composing lights + doors reply. No tool call needed."
            )
            if not on_r:
                l_part = "All lights are off."
            elif len(on_r) == len(avail_r):
                l_part = "All lights are on."
            else:
                l_part = f"Lights on: {room_names(on_r)}. Lights off: {room_names(off_r)}."
            if not unlocked:
                d_part = "All doors are locked."
            elif len(unlocked) == len(avail_d):
                d_part = "All doors are unlocked."
            else:
                d_part = (
                    f"Unlocked: {door_names(unlocked)}. "
                    f"Locked: {door_names(locked)}."
                )
            resp = f"{l_part} {d_part}"
 
        # ── MULTI: lights + TV ──────────────────────────────────────────
        elif cat == "multi_lights_tv":
            if not tv_d: continue
            on_tvs = [r for r in tv_d if tv_d[r] == "on"]
            prompt = random.choice(MULTI_LIGHTS_TV_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: lights={{{l_ss}}}, tv={{{tv_ss}}}. "
                f"Lights on: {on_r}. TVs on: {on_tvs}. "
                f"Composing lights + TV reply. No tool call needed."
            )
            l_part = (
                "All lights are off." if not on_r
                else f"Lights on: {room_names(on_r)}."
            )
            tv_part = (
                "No TVs are on." if not on_tvs
                else (f"The {ROOM_DISPLAY[on_tvs[0]]} TV is on."
                      if len(on_tvs) == 1
                      else f"TVs on: {room_names(on_tvs)}.")
            )
            resp = f"{l_part} {tv_part}"
 
        # ── MULTI: doors + scene ────────────────────────────────────────
        elif cat == "multi_doors_scene":
            prompt = random.choice(MULTI_DOORS_SCENE_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: doors={{{d_ss}}}, active_scene={scene or 'none'}. "
                f"Locked: {locked}. Unlocked: {unlocked}. "
                f"Composing doors + scene reply. No tool call needed."
            )
            if len(locked) == len(avail_d):
                d_part = "All doors are locked."
            elif not locked:
                d_part = "All doors are unlocked."
            else:
                d_part = (
                    f"Locked: {door_names(locked)}. "
                    f"Unlocked: {door_names(unlocked)}."
                )
            sc_part = (
                f"The {scene.replace('_', ' ').title()} scene is active."
                if scene else "No scene is active."
            )
            resp = f"{d_part} {sc_part}"
 
        # ── MULTI: thermostat + scene ───────────────────────────────────
        elif cat == "multi_temp_scene":
            prompt = random.choice(MULTI_TEMP_SCENE_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: thermostat={t}F/{m}, active_scene={scene or 'none'}. "
                f"Composing thermostat + scene reply. No tool call needed."
            )
            sc_part = (
                f"The {scene.replace('_', ' ').title()} scene is active."
                if scene else "No scene is active."
            )
            resp = f"Thermostat: {t}°F in {m} mode. {sc_part}"
 
        # ── MULTI: TV + speaker ─────────────────────────────────────────
        elif cat == "multi_tv_speaker":
            if not tv_d or not sp_d: continue
            on_tvs  = [r for r in tv_d if tv_d[r] == "on"]
            playing = [r for r in sp_d if sp_d[r] == "playing"]
            paused  = [r for r in sp_d if sp_d[r] == "paused"]
            prompt = random.choice(MULTI_TV_SPEAKER_Q)
            think = (
                f"User asks '{prompt}'. "
                f"Reading STATE: tv={{{tv_ss}}}, speaker={{{sp_ss}}}. "
                f"TVs on: {on_tvs}. Playing: {playing}. Paused: {paused}. "
                f"Composing TV + speaker reply. No tool call needed."
            )
            tv_part = (
                "No TVs are on." if not on_tvs
                else (f"The {ROOM_DISPLAY[on_tvs[0]]} TV is on."
                      if len(on_tvs) == 1
                      else f"TVs on: {room_names(on_tvs)}.")
            )
            if playing:
                sp_part = (
                    f"The {room_names(playing)} speaker is playing."
                    if len(playing) == 1 else
                    f"Speakers playing: {room_names(playing)}."
                )
            elif paused:
                sp_part = (
                    f"The {room_names(paused)} speaker is paused."
                    if len(paused) == 1 else
                    f"Speakers paused: {room_names(paused)}."
                )
            else:
                sp_part = "No speakers are playing."
            resp = f"{tv_part} {sp_part}"
 
        # ── FULL HOME STATUS ────────────────────────────────────────────
        else:  # full_home
            prompt = random.choice(FULL_HOME_Q)
            think = (
                f"User asks '{prompt}'. Full home status requested. "
                f"Reading ALL STATE: "
                f"lights={{{l_ss}}}, "
                f"doors={{{d_ss}}}, "
                f"thermostat={t}F/{m}, "
                f"scene={scene or 'none'}, "
                f"tv={{{tv_ss}}}, "
                f"speaker={{{sp_ss}}}, "
                f"fan={{{fan_ss}}}. "
                f"Composing full home report. No tool call needed."
            )
 
            # — lights —
            if not on_r:
                l_part = "All lights are off."
            elif len(on_r) == len(avail_r):
                l_part = "All lights are on."
            else:
                l_part = (
                    f"Lights on: {room_names(on_r)}. "
                    f"Lights off: {room_names(off_r)}."
                )
 
            # — doors —
            if len(locked) == len(avail_d):
                d_part = "All doors are locked."
            elif not locked:
                d_part = "All doors are unlocked."
            else:
                d_part = (
                    f"Locked: {door_names(locked)}. "
                    f"Unlocked: {door_names(unlocked)}."
                )
 
            # — thermostat —
            th_part = f"Thermostat: {t}°F in {m} mode."
 
            # — scene —
            sc_part = (
                f"Scene: {scene.replace('_', ' ').title()}."
                if scene else "No scene active."
            )
 
            # — TV —
            if tv_d:
                on_tvs  = [r for r in tv_d if tv_d[r] == "on"]
                off_tvs = [r for r in tv_d if tv_d[r] == "off"]
                if not on_tvs:
                    tv_part = "TV: off."
                elif len(on_tvs) == 1:
                    tv_part = f"TV: {ROOM_DISPLAY[on_tvs[0]]} is on."
                else:
                    tv_part = f"TVs on: {room_names(on_tvs)}."
            else:
                tv_part = ""
 
            # — speaker —
            if sp_d:
                playing = [r for r in sp_d if sp_d[r] == "playing"]
                paused  = [r for r in sp_d if sp_d[r] == "paused"]
                if playing:
                    sp_part = (
                        f"Speaker: {room_names(playing)} is playing."
                        if len(playing) == 1 else
                        f"Speakers playing: {room_names(playing)}."
                    )
                elif paused:
                    sp_part = (
                        f"Speaker: {room_names(paused)} is paused."
                        if len(paused) == 1 else
                        f"Speakers paused: {room_names(paused)}."
                    )
                else:
                    sp_part = "Speaker: stopped."
            else:
                sp_part = ""
 
            # — fan —
            if fan_d:
                on_fans  = [r for r in fan_d if fan_d[r]["state"] == "on"]
                off_fans = [r for r in fan_d if fan_d[r]["state"] == "off"]
                if not on_fans:
                    fan_part = "Fans: all off."
                else:
                    fan_items = [
                        f"{ROOM_DISPLAY[r]} ({fan_d[r]['speed']})" for r in on_fans
                    ]
                    fan_part = "Fans on: " + ", ".join(fan_items) + "."
                    if off_fans:
                        fan_part += f" Fans off: {room_names(off_fans)}."
            else:
                fan_part = ""
 
            segments = [l_part, d_part, th_part, sc_part]
            if tv_part:  segments.append(tv_part)
            if sp_part:  segments.append(sp_part)
            if fan_part: segments.append(fan_part)
            resp = " ".join(segments)
 
        # ── all categories share the same build_ex call ─────────────────
        if not resp or not prompt:
            continue
 
        examples.append(build_ex(
            prompt, [], resp,
            avail_r, avail_d, state,
            user_room=user_room,
            action_log=act_log,
            think_trace=think,
            category="state_report_queries",
        ))
 
    return examples[:target]


def gen_action_log_lights(target: int = 2_000) -> list:
    """GAP F reinforcement: pronoun traces state current AND target explicitly."""
    examples = []
    undo_p      = ["Undo that.", "Revert that.", "Take that back.", "Actually undo that."]
    rep_t       = ["Do the same for the {r}.", "Same for {r}.", "Also do the {r}."]
    again_p     = ["Do that again.", "Again.", "Repeat that.", "One more time."]
    pronoun_off = ["Turn it off.", "Kill it.", "Shut it off.", "Off it."]
    pronoun_on  = ["Turn it on.", "Switch it on.", "Put it back on.", "On it."]

    for _ in range(target):
        avail_r, avail_d = sample_topology()
        r1  = random.choice(avail_r)
        r2  = random.choice([r for r in avail_r if r != r1]) if len(avail_r) > 1 else r1
        s   = random.choice(["on", "off"])
        opp = "off" if s == "on" else "on"
        primary_mins = random.randint(1, 4)
        primary_txn  = fmt_txn(primary_mins,
                               [f"toggle_lights(room={r1}, state={s})"],
                               f"{ROOM_DISPLAY[r1]} light turned {s}.")
        if random.random() < 0.6:
            dist = build_distractor_log(avail_r, avail_d,
                                        n=random.randint(1, 2),
                                        start_mins=primary_mins + random.randint(7, 14))
            log = primary_txn + "\n" + dist
        else:
            log = primary_txn
        state = generate_random_state(avail_r, avail_d)
        apply_force(state, {"lights": {r1: s}}, avail_r, avail_d)
        mode    = random.choice(["undo", "repeat", "again", "pronoun"])
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        # NOISE for all modes — action log resolution ignores user location
        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f" (ignoring current_user_room='{noise_room}')" if noise_room else " (user room is unknown)"

        if mode == "undo":
            think = (
                f"User said 'undo'. "
                f"RECENT ACTIONS: first block ({t_label}): {ROOM_DISPLAY[r1]} light → {s}. "
                f"Undo: reversing to {opp}."
            )
            examples.append(build_ex(random.choice(undo_p),
                [{"name": "toggle_lights", "args": {"room": r1, "state": opp}}],
                f"{ROOM_DISPLAY[r1].title()} light is now {opp}.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_lights"))

        elif mode == "repeat" and r2 != r1:
            apply_force(state, {"lights": {r2: opp}}, avail_r, avail_d)
            think = (
                f"User said 'same for {ROOM_DISPLAY[r2]}'. "
                f"RECENT ACTIONS: first block ({t_label}): {ROOM_DISPLAY[r1]} → {s}. "
                f"'Same for {ROOM_DISPLAY[r2]}' → toggle_lights(room={r2}, state={s})."
            )
            examples.append(build_ex(random.choice(rep_t).format(r=ROOM_DISPLAY[r2]),
                [{"name": "toggle_lights", "args": {"room": r2, "state": s}}],
                f"{ROOM_DISPLAY[r2].title()} light turned {s}.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_lights"))

        elif mode == "pronoun":
            current_state = s
            user_wants    = opp
            phrase = random.choice(pronoun_off if user_wants == "off" else pronoun_on)
            think = (
                f"User said '{phrase}'. "
                f"Pronoun 'it' → first [...] block ({t_label}): {ROOM_DISPLAY[r1]} light. "
                f"Current state: '{current_state}'. "
                f"User wants: '{user_wants}' (opposite of current). "
                f"Calling toggle_lights(room={r1}, state={user_wants})."
            )
            examples.append(build_ex(phrase,
                [{"name": "toggle_lights", "args": {"room": r1, "state": user_wants}}],
                f"The {ROOM_DISPLAY[r1]} light is now {user_wants}.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_lights"))

        else:  # again
            apply_force(state, {"lights": {r1: opp}}, avail_r, avail_d)
            think = (
                f"User said 'again'. "
                f"RECENT ACTIONS: first block ({t_label}): {ROOM_DISPLAY[r1]} → {s}. "
                f"'Again' → state currently {opp}, toggling back to {s}."
            )
            examples.append(build_ex(random.choice(again_p),
                [{"name": "toggle_lights", "args": {"room": r1, "state": s}}],
                f"{ROOM_DISPLAY[r1].title()} light turned {s} again.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_lights"))

    return examples


def gen_action_log_doors(target: int = 1_500) -> list:
    examples = []
    undo_p     = ["Undo that.", "Revert that.", "Actually undo."]
    pronoun_lk = ["Lock it.", "Close it.", "Secure it."]
    pronoun_ul = ["Unlock it.", "Open it.", "Open it up."]
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        d1  = random.choice(avail_d)
        s   = random.choice(["lock", "unlock"])
        opp = "unlock" if s == "lock" else "lock"
        aw  = "locked" if s == "lock" else "unlocked"
        primary_mins = random.randint(1, 5)
        primary_txn  = fmt_txn(primary_mins,
                               [f"lock_door(door={d1}, state={s})"],
                               f"{DOOR_DISPLAY[d1]} {aw}.")
        if random.random() < 0.6:
            dist = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2),
                                        start_mins=primary_mins + random.randint(7, 14))
            log = primary_txn + "\n" + dist
        else:
            log = primary_txn
        state = generate_random_state(avail_r, avail_d)
        apply_force(state, {"doors": {d1: aw}}, avail_r, avail_d)
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        # NOISE for both modes — action log resolution ignores user location
        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f" (ignoring current_user_room='{noise_room}')" if noise_room else " (user room is unknown)"

        if random.random() < 0.6:
            think = (
                f"User said 'undo'. "
                f"RECENT ACTIONS: first block ({t_label}): "
                f"lock_door({DOOR_DISPLAY[d1]}, {s}) → {aw}. "
                f"Undo: reverting to '{opp}'."
            )
            examples.append(build_ex(random.choice(undo_p),
                [{"name": "lock_door", "args": {"door": d1, "state": opp}}],
                f"{DOOR_DISPLAY[d1].title()} door is now {opp}ed.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_doors"))
        else:
            phrase = random.choice(pronoun_ul if s == "lock" else pronoun_lk)
            think = (
                f"User said '{phrase}'. "
                f"Command targets the first [...] block ({t_label}): {DOOR_DISPLAY[d1]}. "
                f"Current state: {aw}. Reversing to {opp}."
            )
            examples.append(build_ex(phrase,
                [{"name": "lock_door", "args": {"door": d1, "state": opp}}],
                f"The {DOOR_DISPLAY[d1]} is now {opp}ed.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room,  # FIX: was set but not passed previously
                think_trace=think, category="action_log_doors"))
    return examples


def gen_action_log_scenes_therm(target: int = 1_000) -> list:
    examples = []
    undo_p  = ["Undo that.", "Revert that.", "Cancel that.", "Go back."]
    again_p = ["Do that again.", "Activate it again.", "Same scene please."]

    for _ in range(target // 2):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        scene = random.choice(SCENES)
        primary_mins = random.randint(1, 5)
        primary_txn  = fmt_txn(primary_mins,
                               [f"set_scene(scene={scene})"],
                               f"{scene.replace('_', ' ').title()} activated.")
        if random.random() < 0.5:
            dist = build_distractor_log(avail_r, avail_d, n=1,
                                        start_mins=primary_mins + random.randint(8, 15))
            log = primary_txn + "\n" + dist
        else:
            log = primary_txn
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        # NOISE — action log resolution ignores user location
        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f" (ignoring current_user_room='{noise_room}')" if noise_room else " (user room is unknown)"

        if random.random() < 0.5:
            think = (
                f"User said 'undo'. "
                f"First block ({t_label}): set_scene({scene}). "
                f"Scenes have no inverse tool. Informing user."
            )
            examples.append(build_ex(random.choice(undo_p), [],
                "I can't undo a scene directly — let me know what to revert.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_scenes"))
        else:
            think = (
                f"User said 'again'. "
                f"First block ({t_label}): set_scene({scene}). "
                f"'Again' → repeating set_scene(scene={scene})."
            )
            examples.append(build_ex(random.choice(again_p),
                [{"name": "set_scene", "args": {"scene": scene}}],
                f"{scene.replace('_', ' ').title()} scene re-activated.",
                avail_r, avail_d, state, action_log=log,
                user_room=noise_room, think_trace=think, category="action_log_scenes"))

    for _ in range(target // 2):
        avail_r, avail_d = sample_topology()
        temp_new = random.randint(65, 78)
        temp_old = random.randint(MIN_T, MAX_T)
        while temp_old == temp_new: temp_old = random.randint(MIN_T, MAX_T)
        mode  = random.choice(["heat", "cool", "auto"])
        state = generate_random_state(avail_r, avail_d)
        apply_force(state, {"thermostat": {"temperature": temp_new, "mode": mode}},
                    avail_r, avail_d)
        primary_mins = random.randint(1, 5)
        log = fmt_txn(primary_mins,
                      [f"set_thermostat(temperature={temp_new}, mode={mode})"],
                      f"Thermostat changed from {temp_old}F to {temp_new}F.")
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f" (ignoring current_user_room='{noise_room}')" if noise_room else " (user room is unknown)"
        think = (
            f"User said 'undo'."
            f"First block ({t_label}): thermostat → {temp_new}F from {temp_old}F. "
            f"Undo: reverting to {temp_old}F."
        )
        examples.append(build_ex(random.choice(undo_p),
            [{"name": "set_thermostat", "args": {"temperature": temp_old, "mode": mode}}],
            f"Thermostat reverted to {temp_old}°F.",
            avail_r, avail_d, state, action_log=log,
            user_room=noise_room, think_trace=think, category="action_log_scenes"))
    return examples



def gen_scenes(target: int = 1_800) -> list:
    """Single definition including 'already active' branch."""
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state   = generate_random_state(avail_r, avail_d)
        scene   = random.choice(SCENES)
        trigger = random.choice(SCENE_TRIGGERS[scene])
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.5 else ""
        if random.random() < 0.20:
            state["active_scene"] = scene
            u_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{u_room}' is irrelevant" if u_room else "user room is unknown and irrelevant"
            
            # UPDATE THINK TRACES:
            think = (
                f"User requested '{scene}' scene. "
                f"STATE shows active_scene='{scene}'. Already active. "
                f"No tool call needed."
            )
            examples.append(build_ex(trigger.capitalize(), [],
                f"The {scene.replace('_', ' ').title()} scene is already active.",
                avail_r, avail_d, state,
                action_log=action_log,user_room=u_room, think_trace=think, category="scenes"))
        else:
            state["active_scene"] = random.choice(
                [None] + [s for s in SCENES if s != scene])
            u_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{u_room}' is irrelevant" if u_room else "user room is unknown and irrelevant"
            
            # UPDATE THINK TRACES:
            think = (
                f"User requested '{scene}' scene. "
                f"STATE shows active_scene={state.get('active_scene') or 'none'}. "
                f"Calling set_scene(scene='{scene}')."
            )
            examples.append(build_ex(trigger.capitalize(),
                [{"name": "set_scene", "args": {"scene": scene}}],
                SCENE_RESP[scene], avail_r, avail_d, state,
                action_log=action_log,user_room=u_room, think_trace=think, category="scenes"))
    return examples


def gen_thermostat(target: int = 1_800) -> list:
    """GAP J reinforcement: think trace always notes current ≠ new."""
    examples = []
    generic_phrases = ["Set temperature to {v}.", "Make it {v} degrees.",
                       "Thermostat to {v}.", "Set temp to {v}.", "Set it to {v}."]
    ac_phrases   = ["Set the AC to {v}.", "Cool the house to {v}."]
    heat_phrases = ["Turn the heat to {v}.", "Warm it up to {v}."]

    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state    = generate_random_state(avail_r, avail_d)
        cur      = state["thermostat"]["temperature"]
        cur_mode = state["thermostat"]["mode"]
        val      = random.randint(MIN_T, MAX_T)
        req_type = random.choices(["generic", "ac", "heat"], weights=[60, 20, 20])[0]
        if req_type == "ac":
            phrase = random.choice(ac_phrases).format(v=val); mode = "cool"
        elif req_type == "heat":
            phrase = random.choice(heat_phrases).format(v=val); mode = "heat"
        else:
            phrase = random.choice(generic_phrases).format(v=val)
            mode   = "cool" if val < cur else "heat" if val > cur else cur_mode
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""
        if val == cur and mode == cur_mode:
            u_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{u_room}' is irrelevant" if u_room else "user room is unknown and irrelevant"
            
            # UPDATE THINK TRACES:
            think = (
                f"User wants thermostat at {val}F."
                f"Thermostat already at {val}F in {mode} mode. "
                f"No action needed.")
            examples.append(build_ex(phrase, [],
                f"The thermostat is already set to {val}°F in {mode} mode.",
                avail_r, avail_d, state,
                action_log=action_log,user_room=u_room, think_trace=think, category="thermostat"))
        else:
            u_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{u_room}' is irrelevant" if u_room else "user room is unknown and irrelevant"
            
            # UPDATE THINK TRACES:
            think = (
                f"User wants thermostat at {val}F. Current: {cur}F/{cur_mode}. "
                f"Change required. "
                f"Calling set_thermostat(temperature={val}, mode='{mode}')."
            )
            examples.append(build_ex(phrase,
                [{"name": "set_thermostat", "args": {"temperature": val, "mode": mode}}],
                f"Thermostat set to {val}°F in {mode} mode.",
                avail_r, avail_d, state,
                action_log=action_log,user_room=u_room, think_trace=think, category="thermostat"))
    return examples


def gen_rejections(target: int = 1_600) -> list:
    """
    FIX-C: Extended to cover unsupported_device for doors, TVs, and fans.
    FIX-V13: Distractor logs injected so model decouples questions from log reads.
    """
    examples = []
    lt = ["Turn on the {r} light.", "Turn off the {r} light."]
    while len(examples) < target:
        choice = random.choice(["device_light", "device_door", "device_tv",
                         "device_fan", "off_topic", "feature",
                         "implicit_local_door", "implicit_local_door"])

        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        
        # Inject standard transaction logs into rejection scenarios
        act_log = build_distractor_log(avail_r, avail_d, n=1) if random.random() < 0.5 else ""

        if choice == "device_light":
            missing_rooms = [r for r in ALL_ROOMS if r not in avail_r]
            if not missing_rooms: continue
            miss  = random.choice(missing_rooms)
            alias = random.choice(ROOM_ALIASES[miss])
            noise_room = random.choice(["", random.choice(avail_r)])
            
            conn_str = ", ".join(avail_r)
            think = (
                f"User wants {miss} light. "
                f"Checking CONNECTED ROOMS (lights): [{conn_str}]. "
                f"'{miss}' is not in the list. "
                f"Calling intent_unclear(unsupported_device)."
            )
            examples.append(build_ex(random.choice(lt).format(r=alias),
                [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
                f"You don't have a {alias} light connected to me.",
                avail_r, avail_d, state, user_room=noise_room, action_log=act_log,
                think_trace=think, category="rejections"))

        elif choice == "device_door":
            missing_doors = [d for d in ALL_DOORS if d not in avail_d]
            if not missing_doors: continue
            miss_d = random.choice(missing_doors)
            alias  = random.choice(DOOR_ALIASES[miss_d])
            verb   = random.choice(["open", "lock", "unlock", "close"])
            noise_room = random.choice(["", random.choice(avail_r)])
            
            conn_str = ", ".join(avail_d)
            think  = (
                f"User wants to {verb} the {alias}. "
                f"Checking CONNECTED DOORS: [{conn_str}]. "
                f"'{miss_d}' is not in CONNECTED DOORS. "
                f"Calling intent_unclear(unsupported_device)."
            )
            examples.append(build_ex(f"{verb.capitalize()} the {alias}.",
                [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
                f"You don't have a {alias} connected to me.",
                avail_r, avail_d, state, user_room=noise_room, action_log=act_log,
                think_trace=think, category="rejections"))

        elif choice == "implicit_local_door":
            rooms_without_door = [r for r in avail_r if r not in avail_d]
            if not rooms_without_door: continue
            u_room = random.choice(rooms_without_door)
            verb   = random.choice(["close the door", "open the door", "lock the door",
                                     "shut the door", "unlock the door"])
            
            conn_str = ", ".join(avail_d)
            think  = (
                f"User is in '{u_room}'. Said '{verb.capitalize()}.'. "
                f"Generic 'the door' → current_user_room='{u_room}'. "
                f"Checking CONNECTED DOORS: [{conn_str}]. "
                f"'{u_room}' is not in the connected doors list. "
                f"Calling intent_unclear(unsupported_device)."
            )
            examples.append(build_ex(f"{verb.capitalize()}.",
                [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
                f"There's no door connected in the {ROOM_DISPLAY[u_room]}.",
                avail_r, avail_d, state, action_log=act_log,
                user_room=u_room, think_trace=think, category="rejections"))

        elif choice == "device_tv":
            rooms_without_tv = [r for r in avail_r if r not in TV_ROOMS]
            rooms_with_tv = [r for r in avail_r if r in TV_ROOMS]
            if not rooms_without_tv: continue
            r     = random.choice(rooms_without_tv)
            alias = random.choice(ROOM_ALIASES[r])
            noise_room = random.choice(["", random.choice(avail_r)])
            prompt = f"Turn on the {alias} TV."
            
            conn_str = ", ".join(rooms_with_tv)
            think = (
                f"User wants to control the TV in {r}. "
                f"Checking CONNECTED TVs: [{conn_str}]. "
                f"'{r}' is not in the connected TV list. "
                f"Calling intent_unclear(unsupported_device)."
            )
            examples.append(build_ex(prompt,
                [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
                f"There's no TV connected in the {alias}.",
                avail_r, avail_d, state, user_room=noise_room, action_log=act_log,
                think_trace=think, category="rejections"))

        elif choice == "device_fan":
            rooms_without_fan = [r for r in avail_r if r not in FAN_ROOMS]
            rooms_with_fan = [r for r in avail_r if r in FAN_ROOMS]
            if not rooms_without_fan: continue
            r     = random.choice(rooms_without_fan)
            alias = random.choice(ROOM_ALIASES[r])
            noise_room = random.choice(["", random.choice(avail_r)])
            
            conn_str = ", ".join(rooms_with_fan)
            think = (
                f"User wants to control the fan in {r}. "
                f"Checking CONNECTED FANS: [{conn_str}]. "
                f"'{r}' is not in CONNECTED FANS. "
                f"Calling intent_unclear(unsupported_device)."
            )
            examples.append(build_ex(f"Turn on the {alias} fan.",
                [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
                f"There's no fan connected in the {alias}.",
                avail_r, avail_d, state, user_room=noise_room, action_log=act_log,
                think_trace=think, category="rejections"))

        elif choice == "off_topic":
            noise_room = random.choice(["", random.choice(avail_r)])
            think = (
                f"Off-topic question. "
                + "Calling intent_unclear(off_topic)."
            )
            examples.append(build_ex(random.choice(OFF_TOPIC_PHRASES),
                [{"name": "intent_unclear", "args": {"reason": "off_topic"}}],
                "I'm a home assistant — I can only control your connected home devices.",
                avail_r, avail_d, state, user_room=noise_room, action_log=act_log,
                think_trace=think, category="rejections"))

        else:  # feature
            r     = random.choice(avail_r)
            phrase = random.choice(UNSUPPORTED_FEATURE_PHRASES).format(r=ROOM_DISPLAY[r])
            noise_room = random.choice(["", random.choice(avail_r)])
            think  = (
                f"Light dimming and color control are unsupported features. "
                f"Calling intent_unclear(unsupported_feature)."
            )
            examples.append(build_ex(phrase,
                [{"name": "intent_unclear", "args": {"reason": "unsupported_feature"}}],
                "I can't do that — I can only turn lights fully on or off.",
                avail_r, avail_d, state, user_room=noise_room, action_log=act_log,
                think_trace=think, category="rejections"))

    return examples[:target]


def gen_incomplete(target: int = 1_200) -> list:
    examples = []
    lp = ["Turn on the light.", "Turn off the light.", "Lights on.",
          "On the light.", "Off the light."]
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.6 else ""
        think = ("User said 'the light' without specifying a room. "
                 "current_user_room is empty. "
                 "Asking for clarification. "
                 "Calling intent_unclear(incomplete).")
        examples.append(build_ex(random.choice(lp),
            [{"name": "intent_unclear", "args": {"reason": "incomplete"}}],
            "Which room's light would you like me to control?",
            avail_r, avail_d, state, user_room="",
            action_log=action_log, think_trace=think,
            category="incomplete_no_room"))
    return examples


def gen_missing_device(target: int = 1_500) -> list:
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state  = generate_random_state(avail_r, avail_d)
        device, phrases = random.choice(UNSUPPORTED_APPLIANCES)
        phrase = random.choice(phrases)
        resp   = random.choice([
            f"I can't control the {device} — it's not connected to me.",
            f"The {device} isn't one of my connected devices."
        ])
        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f"current_user_room='{noise_room}' is irrelevant" if noise_room else "user room is unknown and irrelevant"
        think = (
            f"User asked about {device}. "
            f"This is not a supported device type. "
            f"Calling intent_unclear(unsupported_device)."
        )
        examples.append(build_ex(phrase,
            [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
            resp, avail_r, avail_d, state,
            user_room=noise_room, think_trace=think, category="missing_device"))
    return examples



def gen_mixed_compound(target: int = 2_000) -> list:
    """Single definition combining always-action and partial-state variants."""
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        choice = random.choice(["light_door", "tv_speaker", "light_fan"])
        partial = random.random() < 0.4
        
        if choice == "tv_speaker":
            tv_rooms = [r for r in avail_r if r in TV_ROOMS]
            sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
            if not tv_rooms or not sp_rooms: continue
            tr, sr = tv_rooms[0], sp_rooms[0]
            avail_r = [r for r in avail_r if r not in TV_ROOMS and r not in SPEAKER_ROOMS] + list(dict.fromkeys([tr, sr]))
            state = generate_random_state(avail_r, avail_d)
        elif choice == "light_fan":
            fan_rooms = [x for x in avail_r if x in FAN_ROOMS]
            if not fan_rooms: continue
            fr = fan_rooms[0]
            r = random.choice([x for x in avail_r if x != fr])
            state = generate_random_state(avail_r, avail_d)
        else:
            state = generate_random_state(avail_r, avail_d)

        calls = []
        think_parts = []
        resp_parts  = []
        prompt      = ""

        if choice == "light_door":
            r = random.choice(avail_r); d = random.choice(avail_d)
            ls_req = random.choice(["on", "off"]); ds_req = random.choice(["lock", "unlock"])
            
            ls_init = ls_req if (partial and random.random() < 0.5) else ("off" if ls_req == "on" else "on")
            ds_init = ds_req if (partial and random.random() < 0.5) else ("unlock" if ds_req == "lock" else "lock")
            aw_init = "locked" if ds_init == "lock" else "unlocked"
            state["lights"][r]["state"] = ls_init
            state["doors"][d] = aw_init
            
            if ls_init == ls_req:
                think_parts.append(f"(1) Checking STATE: {r}={ls_init}. Match — skip")
                resp_parts.append(f"the {ROOM_DISPLAY[r]} light is already {ls_req}")
            else:
                calls.append({"name": "toggle_lights", "args": {"room": r, "state": ls_req}})
                think_parts.append(f"(1) Checking STATE: {r}={ls_init}. Mismatch — toggle_lights({r}, {ls_req})")
                resp_parts.append(f"{ROOM_DISPLAY[r].title()} light {ls_req}")
                
            aw = "locked" if ds_req == "lock" else "unlocked"
            if ds_init == ds_req:
                think_parts.append(f"(2) Checking STATE: {d}={aw_init}. Match — skip")
                resp_parts.append(f"the {DOOR_DISPLAY[d]} is already {aw}")
            else:
                calls.append({"name": "lock_door", "args": {"door": d, "state": ds_req}})
                think_parts.append(f"(2) Checking STATE: {d}={aw_init}. Mismatch — lock_door({d}, {ds_req})")
                resp_parts.append(f"{DOOR_DISPLAY[d].title()} {aw}")
                
            prompt = random.choice([
                f"Turn {ls_req} the {ROOM_DISPLAY[r]} light and {ds_req} the {DOOR_DISPLAY[d]}.",
                f"{ds_req.capitalize()} the {DOOR_DISPLAY[d]} and turn {ls_req} the {ROOM_DISPLAY[r]} light."
            ])

        elif choice == "tv_speaker":
            ts_req = random.choice(["on", "off"]); sa_req = random.choice(["play", "stop"])
            
            ts_init = ts_req if (partial and random.random() < 0.5) else ("off" if ts_req == "on" else "on")
            sp_init = sa_req if (partial and random.random() < 0.5) else ("stop" if sa_req == "play" else "play")
            
            state["tv"][tr] = ts_init
            state["speaker"][sr] = "playing" if sp_init == "play" else "stopped"
            
            if ts_init == ts_req:
                think_parts.append(f"(1) Checking STATE: tv_{tr}={ts_init}. Match — skip")
                resp_parts.append(f"the {ROOM_DISPLAY[tr]} TV is already {ts_req}")
            else:
                calls.append({"name": "control_tv", "args": {"room": tr, "state": ts_req}})
                think_parts.append(f"(1) Checking STATE: tv_{tr}={ts_init}. Mismatch — control_tv({tr}, {ts_req})")
                resp_parts.append(f"{ROOM_DISPLAY[tr].title()} TV {ts_req}")
                
            resp_state = "playing" if sa_req == "play" else "stopped"
            actual_sp_state = state["speaker"][sr]
            if sp_init == sa_req:
                think_parts.append(f"(2) Checking STATE: speaker_{sr}={actual_sp_state}. Match — skip")
                resp_parts.append(f"the {ROOM_DISPLAY[sr]} speaker is already {resp_state}")
            else:
                calls.append({"name": "control_speaker", "args": {"room": sr, "action": sa_req}})
                think_parts.append(f"(2) Checking STATE: speaker_{sr}={actual_sp_state}. Mismatch — control_speaker({sr}, {sa_req})")
                resp_parts.append(f"{ROOM_DISPLAY[sr]} speaker {'now playing' if sa_req=='play' else 'stopped'}")
                
            prompt = f"{'On' if ts_req=='on' else 'Off'} the TV and {'play' if sa_req=='play' else 'stop'} the music."

        else:
            ls_req = random.choice(["on", "off"]); fs_req = random.choice(["on", "off"])
            
            ls_init = ls_req if (partial and random.random() < 0.5) else ("off" if ls_req == "on" else "on")
            fs_init = fs_req if (partial and random.random() < 0.5) else ("off" if fs_req == "on" else "on")
            
            state["lights"][r]["state"] = ls_init
            state["fan"][fr]["state"] = fs_init
            
            if ls_init == ls_req:
                think_parts.append(f"(1) Checking STATE: {r}={ls_init}. Match — skip")
                resp_parts.append(f"the {ROOM_DISPLAY[r]} light is already {ls_req}")
            else:
                calls.append({"name": "toggle_lights", "args": {"room": r, "state": ls_req}})
                think_parts.append(f"(1) Checking STATE: {r}={ls_init}. Mismatch — toggle_lights({r}, {ls_req})")
                resp_parts.append(f"{ROOM_DISPLAY[r].title()} light {ls_req}")
                
            if fs_init == fs_req:
                think_parts.append(f"(2) Checking STATE: fan_{fr}={fs_init}. Match — skip")
                resp_parts.append(f"the {ROOM_DISPLAY[fr]} fan is already {fs_req}")
            else:
                calls.append({"name": "control_fan", "args": {"room": fr, "state": fs_req}})
                think_parts.append(f"(2) Checking STATE: fan_{fr}={fs_init}. Mismatch — control_fan({fr}, {fs_req})")
                resp_parts.append(f"{ROOM_DISPLAY[fr].title()} fan {fs_req}")
                
            prompt = f"Turn {ls_req} the {ROOM_DISPLAY[r]} light and {fs_req} the {ROOM_DISPLAY[fr]} fan."

        think = f"Compound request. Counting sub-actions: {' '.join(think_parts)}."
        
        if len(resp_parts) == 1:
            final_resp = resp_parts[0].capitalize() + "."
        else:
            final_resp = resp_parts[0].capitalize() + " and " + resp_parts[1] + "."

        examples.append(build_ex(prompt, calls, final_resp,
            avail_r, avail_d, state,
            think_trace=think, category="mixed_compound"))

    return examples


def gen_multi_room_lights(target: int = 1_500) -> list:
    """
    Multi-room light control with format variants.
    Added formats fix turns 94, 98, 105, 182:
      - 'the kitchen the bedroom and the office light' (no commas, the-prefix)
      - 'kitchen light, bedroom light, and hallway light' (light after each)
      - 'Kitchen light on. Bedroom light on.' (sentence format)
      - 'kitchen,bathroom,office light' (no-space comma)
    Think trace explicitly states N calls, one per room, no duplicates.
    """
    examples = []
    on_t  = ["On the {r_list} lights.", "Turn on the {r_list} lights.",
              "Switch on the {r_list} lights.", "Lights on in the {r_list}.",
              "Can you turn on {r_list} lights?"]
    off_t = ["Off the {r_list} lights.", "Turn off the {r_list} lights.",
              "Kill the {r_list} lights.", "Switch off the {r_list} lights."]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=5)
        n_rooms = random.choices([2, 3, 4, 5], weights=[30, 30, 25, 15])[0]
        n_rooms = min(n_rooms, len(avail_r))
        if n_rooms < 2: continue
        chosen  = random.sample(avail_r, n_rooms)
        chosen  = list(dict.fromkeys(chosen))  # safety dedup
        state   = generate_random_state(avail_r, avail_d)
        s       = random.choice(["on", "off"])
        opp     = "off" if s == "on" else "on"
        for r in chosen:
            apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)

        aliases = [random.choice(ROOM_ALIASES[r]) for r in chosen]
        calls   = [{"name": "toggle_lights", "args": {"room": r, "state": s}}
                   for r in chosen]   # exactly N calls, no duplicates
        resp    = " ".join(f"{a.title()} light {s}." for a in aliases)

        fmt = random.choices(
            ["standard", "light_after_each", "no_space_comma",
             "no_comma_the", "sentence", "verb_each"],
            weights=[30, 20, 15, 15, 12, 8]
        )[0]

        if fmt == "standard":
            if n_rooms == 2:
                r_list = f"{aliases[0]} and {aliases[1]}"
            else:
                if random.random() < 0.3:           # messy: no space after comma
                    r_list = ",".join(aliases[:-1]) + f" and {aliases[-1]}"
                else:
                    r_list = ", ".join(aliases[:-1]) + f", and {aliases[-1]}"
            tmpl   = random.choice(on_t if s == "on" else off_t)
            prompt = tmpl.format(r_list=r_list)

        elif fmt == "light_after_each":
            # "kitchen light, bedroom light, and office light"
            parts = [f"{a} light" for a in aliases]
            if n_rooms == 2:
                r_list = f"{parts[0]} and {parts[1]}"
            else:
                r_list = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            verb   = "on" if s == "on" else "off"
            prompt = random.choice([
                f"{verb.capitalize()} the {r_list}.",
                f"Turn {verb} the {r_list}.",
            ])

        elif fmt == "no_space_comma":
            # "on the kitchen,bathroom,office light"
            r_list = ",".join(aliases)
            verb   = "on" if s == "on" else "off"
            prompt = random.choice([
                f"{verb.capitalize()} the {r_list} light.",
                f"Turn {verb} the {r_list} light.",
            ])

        elif fmt == "no_comma_the":
            # "the kitchen the bedroom and the office light"
            parts  = [f"the {a}" for a in aliases]
            if n_rooms == 2:
                r_list = f"{parts[0]} and {parts[1]}"
            else:
                r_list = " ".join(parts[:-1]) + f" and {parts[-1]}"
            verb   = "on" if s == "on" else "off"
            prompt = random.choice([
                f"Turn {verb} {r_list} light.",
                f"{verb.capitalize()} {r_list} lights.",
            ])

        elif fmt == "sentence":
            # "Kitchen light on. Bedroom light on. Office light on."
            parts  = [f"{a.title()} light {s}." for a in aliases]
            prompt = " ".join(parts)

        else:  # verb_each
            # "Turn on the kitchen light, turn on the bedroom light, and turn on the office light."
            verb  = "on" if s == "on" else "off"
            parts = [f"turn {verb} the {a} light" for a in aliases]
            if n_rooms == 2:
                prompt = f"{parts[0].capitalize()} and {parts[1]}."
            else:
                prompt = ", ".join(parts[:-1]).capitalize() + f", and {parts[-1]}."

        call_traces = ", ".join(f"toggle_lights({r}, {s})" for r in chosen)
        think = (
            f"User listed {n_rooms} rooms: {', '.join(chosen)}. "
            f"All {n_rooms} must be turned {s}. "
            f"Issuing exactly {n_rooms} separate toggle_lights calls "
            f"(one per room, no duplicates): {call_traces}."
        )
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think, category="multi_room_lights"))

    return examples


def gen_status_queries(target: int = 1_500) -> list:
    examples = []
    while len(examples) < target:
        avail_r, avail_d = sample_topology()
        state  = generate_random_state(avail_r, avail_d)
        choice = random.choice(["light", "door", "thermostat", "scene"])
        if choice == "light":
            r = random.choice(avail_r); s = state["lights"][r]["state"]
            alias = random.choice(ROOM_ALIASES[r])
            noise_room = random.choice(["", r, random.choice(avail_r)])
            loc_str = f"current_user_room='{noise_room}'" if noise_room else "user room is unknown"
            think = (
                f"User asks status of the '{alias}' light . "
                f"Explicit name — checking STATE directly: {r}={s}. "
                f"Providing text reply, no tool call needed."
            )
            q = random.choice([f"Is the {alias} light on?", f"Check the {alias} light."])
            examples.append(build_ex(q, [], f"The {alias} light is {s}.",
                avail_r, avail_d, state, user_room=noise_room,
                think_trace=think, category="status_queries"))
        elif choice == "door":
            d = random.choice(avail_d); s = state["doors"][d]
            alias = random.choice(DOOR_ALIASES[d])
            noise_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{noise_room}'" if noise_room else "user room is unknown"
            think = (
                f"User asks status of the '{alias}' . "
                f"Explicit name — checking STATE directly: {d}={s}. "
                f"Providing text reply, no tool call needed."
            )
            q = random.choice([f"Is the {alias} locked?", f"Check the {alias}."])
            examples.append(build_ex(q, [], f"The {alias} is {s}.",
                avail_r, avail_d, state, user_room=noise_room,
                think_trace=think, category="status_queries"))
        elif choice == "thermostat":
            t, m = state["thermostat"]["temperature"], state["thermostat"]["mode"]
            noise_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{noise_room}' is irrelevant" if noise_room else "user room is unknown and irrelevant"
            think = (
                f"User asks thermostat status."
                f"STATE shows {t}F/{m}. Providing text reply."
            )
            examples.append(build_ex("What's the temperature?",
                [], f"The thermostat is set to {t}°F in {m} mode.",
                avail_r, avail_d, state, user_room=noise_room,
                think_trace=think, category="status_queries"))
        else:
            scene = state.get("active_scene")
            noise_room = random.choice(["", random.choice(avail_r)])
            loc_str = f"current_user_room='{noise_room}' is irrelevant" if noise_room else "user room is unknown and irrelevant"
            think = (
                f"User asks active scene."
                f"STATE shows {scene}. Providing text reply."
            )
            resp  = (f"{scene.replace('_', ' ').title()} is active."
                     if scene else "No scene is currently active.")
            examples.append(build_ex("What scene is active?", [], resp,
                avail_r, avail_d, state, user_room=noise_room,
                think_trace=think, category="status_queries"))
    return examples


def gen_room_priority_over_log(target: int = 2_500) -> list:
    """GAP A critical: current_user_room always beats action log."""
    examples = []
    on_p  = ["Turn the light on.", "Lights on.", "On the light.", "On this light.",
              "Switch on the light.", "Light on please.", "Can you turn on the light?"]
    off_p = ["Turn the light off.", "Lights off.", "Off the light.", "Off this light.",
              "Switch off the light.", "Light off please.", "Kill the light."]
    ul_p  = ["Open the door.", "Unlock this door.", "Open this door."]
    lk_p  = ["Close the door.", "Lock this door.", "Close this door."]

    for _ in range(target):
        choice = random.choice(["light", "door"])
        if choice == "light":
            cur_room = random.choice(ALL_ROOMS)
            log_room = random.choice([r for r in ALL_ROOMS if r != cur_room])
            avail_r, avail_d = sample_topology(required_rooms=[cur_room, log_room])
            state = generate_random_state(avail_r, avail_d)
            s   = random.choice(["on", "off"])
            opp = "off" if s == "on" else "on"
            apply_force(state, {"lights": {cur_room: opp}}, avail_r, avail_d)
            log_mins    = random.randint(3, 10)
            primary_txn = get_cross_room_txn(avail_r, avail_d, cur_room, log_mins)
            if random.random() < 0.5:
                dist = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2),
                                            start_mins=log_mins + random.randint(8, 15))
                action_log = primary_txn + "\n" + dist
            else:
                action_log = primary_txn
            alias  = random.choice(ROOM_ALIASES[cur_room])
            prompt = random.choice(on_p if s == "on" else off_p)
            think  = (
                f"User is in '{cur_room}'. Said '{prompt}'. "
                f"Generic 'the light' resolves to current_user_room='{cur_room}'. "
                f"State shows {cur_room}:{opp}, user wants {s}. "
                f"Calling toggle_lights(room={cur_room}, state={s})."
            )
            examples.append(build_ex(prompt,
                [{"name": "toggle_lights", "args": {"room": cur_room, "state": s}}],
                f"The {alias} light is now {s}.",
                avail_r, avail_d, state,
                user_room=cur_room, action_log=action_log,
                think_trace=think, category="room_priority_over_log"))

        else:
            door_choices = ["bedroom", "bathroom", "office", "kitchen", "living_room"]
            cur_door = random.choice(door_choices)
            avail_r, avail_d = sample_topology(required_doors=[cur_door])
            state = generate_random_state(avail_r, avail_d)
            s   = random.choice(["lock", "unlock"])
            opp = "unlock" if s == "lock" else "lock"
            aw  = "locked" if s == "lock" else "unlocked"
            apply_force(state, {"doors": {cur_door: opp}}, avail_r, avail_d)
            log_r    = random.choice(avail_r)
            log_ls   = random.choice(["on", "off"])
            log_mins = random.randint(3, 12)
            primary_txn = fmt_txn(log_mins,
                                  [f"toggle_lights(room={log_r}, state={log_ls})"],
                                  f"{ROOM_DISPLAY[log_r]} light turned {log_ls}.")
            action_log = primary_txn + (
                "\n" + build_distractor_log(avail_r, avail_d, n=1,
                                            start_mins=log_mins + random.randint(8, 15))
                if random.random() < 0.4 else "")
            alias  = random.choice(DOOR_ALIASES[cur_door])
            prompt = random.choice(ul_p if s == "unlock" else lk_p)
            think  = (
                f"User in '{cur_door}'. Said '{prompt}'. "
                f"Generic 'the door' resolves to current_user_room='{cur_door}'. "
                f"Calling lock_door(door={cur_door}, state={s})."
            )
            examples.append(build_ex(prompt,
                [{"name": "lock_door", "args": {"door": cur_door, "state": s}}],
                f"The {alias} is now {aw}.",
                avail_r, avail_d, state,
                user_room=cur_door, action_log=action_log,
                think_trace=think, category="room_priority_over_log"))
    return examples


def gen_all_devices_with_room_set(target: int = 1_500) -> list:
    examples = []
    on_p  = ["Turn on all the lights.", "All lights on.", "On all the lights.",
              "Every light on.", "Lights on everywhere.", "Switch on all the lights.",
              "All the lights please.", "Can you turn on all lights?"]
    off_p = ["Turn off all the lights.", "All lights off.", "Off all the lights.",
              "Every light off.", "Lights off everywhere.", "Kill all the lights.",
              "All lights off please."]
    lk_p  = ["Close all the doors.", "Lock all doors.", "Secure every door.",
              "Lock all the doors.", "Close every door.", "Close all doors.",
              "Secure all the doors.", "Lock every door.", "Close up all the doors."]
    ul_p  = ["Open all the doors.", "Unlock all doors.", "Open every door.",
              "Unlock all the doors.", "Open up all doors.", "Unlock every door."]
 
    for _ in range(target):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        choice = random.choice(["light", "door"])
        action_log = ""          # ← ADD THIS LINE; safe default before any branch
        if choice == "light":
            s   = random.choice(["on", "off"])
            opp = "off" if s == "on" else "on"
            action_rooms = [r2 for r2 in avail_r if state["lights"][r2]["state"] == opp]
            if not action_rooms:
                summary = ", ".join(
                    f"{ROOM_DISPLAY[r2]}:{state['lights'][r2]['state']}" for r2 in avail_r)
                already_state = s   # all lights already in the requested state
                prompt = random.choice(on_p if s == "on" else off_p)
                think = (
                    f"User said '{prompt}'. Global light scope. "
                    f"Checking ALL connected lights: {summary}. "
                    f"Result: ALL lights already {already_state}. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt, [],
                    f"All lights are already {already_state}.",
                    avail_r, avail_d, state,
                    user_room=r, action_log=action_log,
                    think_trace=think,
                    category="all_devices_with_room_set"))
                continue
            summary    = ", ".join(f"{ROOM_DISPLAY[r2]}:{state['lights'][r2]['state']}" for r2 in avail_r)
            room_names = ", ".join(ROOM_DISPLAY[r2] for r2 in action_rooms)
            dist_mins  = random.randint(6, 15)
            action_log = get_cross_room_txn(avail_r, avail_d, r, dist_mins)
            if random.random() < 0.4:
                more = build_distractor_log(avail_r, avail_d, n=1,
                                            start_mins=dist_mins + random.randint(8, 15))
                action_log = action_log + "\n" + more
            prompt = random.choice(on_p if s == "on" else off_p)
            think = (
                f"User said '{prompt}'. Single-device bulk command — lights only. "
                f"Global light scope. Checking ALL connected lights: {summary}. "
                f"{len(action_rooms)} light(s) need to turn {s} ({room_names}). "
                f"Issuing {len(action_rooms)} toggle_lights(state={s}) calls individually."
            )
            calls = [{"name": "toggle_lights", "args": {"room": r2, "state": s}} for r2 in action_rooms]
            resp  = " ".join(f"{ROOM_DISPLAY[r2].title()} light {s}." for r2 in action_rooms)
        else:
            s      = random.choice(["lock", "unlock"])
            aw     = "locked" if s == "lock" else "unlocked"
            opp_aw = "unlocked" if s == "lock" else "locked"
            action_doors = [d for d in avail_d if state["doors"][d] == opp_aw]
            if not action_doors:
                summary = ", ".join(
                    f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
                prompt = random.choice(lk_p if s == "lock" else ul_p)
                think = (
                    f"User said '{prompt}'. Global door scope. "
                    f"Checking ALL connected doors: {summary}. "
                    f"Result: ALL doors already {aw}. "
                    f"State already matches request. No tool calls needed."
                )
                examples.append(build_ex(prompt, [],
                    f"All doors are already {aw}.",
                    avail_r, avail_d, state,
                    user_room=r, action_log=action_log,
                    think_trace=think,
                    category="all_devices_with_room_set"))
                continue
            summary    = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            door_names = ", ".join(DOOR_DISPLAY[d] for d in action_doors)
            action_log = build_distractor_log(avail_r, avail_d, n=1) if random.random() < 0.4 else ""
            prompt = random.choice(lk_p if s == "lock" else ul_p)
            think = (
                f"User said '{prompt}'. Single-device bulk command — doors only. "
                f"Global door scope. Checking ALL connected doors: {summary}. "
                f"{len(action_doors)} door(s) need to turn {aw} ({door_names}). "
                f"Issuing {len(action_doors)} lock_door(state={s}) calls individually."
            )
            calls = [{"name": "lock_door", "args": {"door": d, "state": s}} for d in action_doors]
            resp  = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in action_doors)
 
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=r, action_log=action_log, think_trace=think,
            category="all_devices_with_room_set"))
 
    return examples


def gen_compound_scene_device(target: int = 3_500) -> list:
    """GAP D: scene + device compound — both tools must be called.
    user_room is varied for tv/speaker so Rule 1 in non-device rooms is trained."""
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        scene = random.choice(SCENES)
        strig = random.choice(SCENE_TRIGGERS[scene])
        device_type = random.choice(["light", "door", "tv", "speaker"])
        user_room_ex = ""   # default; overridden per-branch where useful

        if device_type == "light":
            r = random.choice(avail_r); alias = random.choice(ROOM_ALIASES[r])
            ls = random.choice(["on", "off"])
            apply_force(state, {"lights": {r: ("off" if ls=="on" else "on")}}, avail_r, avail_d)
            prompts = [
                f"Set scene to {strig} and turn {ls} the {alias} light.",
                f"{strig.capitalize()} please and {ls} the {alias} light.",
                f"Activate {strig} and {ls} the {alias} lights.",
            ]
            calls = [{"name": "set_scene", "args": {"scene": scene}},
                     {"name": "toggle_lights", "args": {"room": r, "state": ls}}]
            resp  = f"{SCENE_RESP[scene]} {alias.title()} light {ls}."
            think = (f"Two sub-actions: (1) set_scene(scene={scene}), "
                     f"(2) toggle_lights(room={r}, state={ls}). Both must be called.")

        elif device_type == "door":
            d = random.choice(avail_d); da = random.choice(DOOR_ALIASES[d])
            ds = random.choice(["lock", "unlock"])
            apply_force(state, {"doors": {d: ("unlocked" if ds=="lock" else "locked")}},
                        avail_r, avail_d)
            aw = "locked" if ds == "lock" else "unlocked"
            prompts = [
                f"Set scene to {strig} and {ds} the {da}.",
                f"{strig.capitalize()} and {ds} the {da}.",
            ]
            calls = [{"name": "set_scene", "args": {"scene": scene}},
                     {"name": "lock_door", "args": {"door": d, "state": ds}}]
            resp  = f"{SCENE_RESP[scene]} {da.title()} {aw}."
            think = (f"Two sub-actions: (1) set_scene(scene={scene}), "
                     f"(2) lock_door(door={d}, state={ds}). Both must be called.")

        elif device_type == "tv":
            tv_rooms = [r for r in avail_r if r in TV_ROOMS]
            if not tv_rooms: continue
            
            # FIX: Force topology to only have ONE TV so implicit "the TV" is legally valid
            avail_r = [r for r in avail_r if r not in TV_ROOMS] + [tv_rooms[0]]
            state = generate_random_state(avail_r, avail_d) # Regen state with new topology
            
            tr = tv_rooms[0]; ta = random.choice(ROOM_ALIASES[tr])
            ts = random.choice(["on", "off"])
            state["tv"][tr] = "off" if ts == "on" else "on"
            non_tv = [r for r in avail_r if r not in TV_ROOMS]
            if non_tv and random.random() < 0.4:
                user_room_ex = random.choice(non_tv)
            prompts = [f"Set scene to {strig} and turn {ts} the {ta} TV.",
                       f"{strig.capitalize()} and {ts} the TV."]
            calls = [{"name": "set_scene", "args": {"scene": scene}},
                     {"name": "control_tv", "args": {"room": tr, "state": ts}}]
            resp  = f"{SCENE_RESP[scene]} {ta.title()} TV {ts}."
            think = (f"Two sub-actions: (1) set_scene(scene={scene}), "
                     f"(2) control_tv(room={tr}, state={ts}). Both must be called.")

        else:  # speaker
            sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
            if not sp_rooms: continue
            
            # FIX: Force topology to only have ONE Speaker so implicit "the speaker" is legally valid
            avail_r = [r for r in avail_r if r not in SPEAKER_ROOMS] + [sp_rooms[0]]
            state = generate_random_state(avail_r, avail_d) # Regen state with new topology
            
            sr = sp_rooms[0]; sa_a = random.choice(ROOM_ALIASES[sr])
            sa = random.choice(["play", "stop"])
            state["speaker"][sr] = "stopped" if sa == "play" else "playing"
            non_sp = [r for r in avail_r if r not in SPEAKER_ROOMS]
            if non_sp and random.random() < 0.4:
                user_room_ex = random.choice(non_sp)
            prompts = [f"Set scene to {strig} and play some music.",
                       f"{strig.capitalize()} and {'play' if sa=='play' else 'stop'} the speaker."]
            calls = [{"name": "set_scene", "args": {"scene": scene}},
                     {"name": "control_speaker", "args": {"room": sr, "action": sa}}]
            resp  = (f"{SCENE_RESP[scene]} "
                     f"{'Playing' if sa=='play' else 'Stopped'} music on the {sa_a} speaker.")
            think = (f"Two sub-actions: (1) set_scene(scene={scene}), "
                     f"(2) control_speaker(room={sr}, action={sa}). Both must be called.")

        # ── shared tail — outside ALL device_type branches ─────────────
        action_log = build_distractor_log(avail_r, avail_d, n=1,
                                          start_mins=random.randint(10, 20)) \
            if random.random() < 0.4 else ""
        examples.append(build_ex(random.choice(prompts), calls, resp,
            avail_r, avail_d, state,
            user_room=user_room_ex, action_log=action_log, think_trace=think,
            category="compound_scene_device"))
    return examples

def gen_undo_multi_action(target: int = 1_200) -> list:
    """
    GAP E (extended): Undo after multi-device transaction.
    Key addition: think trace now explicitly states 'them'/'those' refers to
    the FIRST [...] block, NOT all currently on lights — prevents state-aware
    path from winning. Also adds 'those lights back' phrases.
    """
    on_back_p = [
        "On them back.", "Turn them back on.", "Put them back on.",
        "Undo all that.", "On those lights back.", "Turn those lights back on.",
        "Turn those back on.", "Put those back on.", "Reverse that.",
        "Bring the lights back.", "Bring them back on.", "Lights back on.",
        "On those back.",
    ]
    off_back_p = [
        "Off them all.", "Turn them all off.", "Kill those lights.",
        "Off those lights.", "Undo that.", "Turn those off.",
        "Off them.", "Put them all off.", "Reverse that.", "Lights off again.",
        "Off those.", "Turn off those lights.",
    ]
    examples = []

    for _ in range(target):
        n_rooms = random.choices([2, 3, 4, 5], weights=[15, 25, 35, 25])[0]
        avail_r, avail_d = sample_topology(min_rooms=max(n_rooms, 2))
        rooms = random.sample(avail_r, min(n_rooms, len(avail_r)))
        if len(rooms) < 2: continue
        state = generate_random_state(avail_r, avail_d)

        prior_state  = random.choice(["on", "off"])
        target_state = "off" if prior_state == "on" else "on"
        
        # ABSOLUTE GROUNDING: Force current state to equal the logged prior state 
        # so reversing it to target_state is physically necessary.
        for r in rooms:
            apply_force(state, {"lights": {r: prior_state}}, avail_r, avail_d)

        primary_mins = random.randint(1, 4)
        call_strs    = [f"toggle_lights(room={r}, state={prior_state})" for r in rooms]
        
        # FIX: Match backend period-separated sentence aggregation for counting accuracy
        summary = " ".join(f"{ROOM_DISPLAY[r].title()} light turned {prior_state}." for r in rooms)
        
        primary_txn = fmt_txn(primary_mins, call_strs, summary)

        n_dist      = random.randint(1, 3)
        distractors = build_distractor_log(avail_r, avail_d, n=n_dist,
                                           start_mins=primary_mins + random.randint(6, 12))
        action_log  = primary_txn + "\n" + distractors

        prompt = random.choice(on_back_p if target_state == "on" else off_back_p)
        calls  = [{"name": "toggle_lights", "args": {"room": r, "state": target_state}}
                  for r in rooms]
        resp   = " ".join(f"{ROOM_DISPLAY[r].title()} light {target_state}." for r in rooms)
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        rooms_str = ", ".join(ROOM_DISPLAY[r] for r in rooms)

        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f" (ignoring current_user_room='{noise_room}')" if noise_room else " (user room is unknown)"
        
        think = (
            f"User said '{prompt}'. "
            f"'back'/'them'/'those'/'undo' → look at the FIRST [...] block in RECENT ACTIONS. "
            f"First block ({t_label}): {len(rooms)} lights ({rooms_str}) were all turned {prior_state}. "
            f"Reversing by turning all {target_state}. "
            f"Issuing {len(rooms)} toggle_lights call(s): "
            + ", ".join(f"toggle_lights({r}, {target_state})" for r in rooms) + "."
        )
        
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, user_room=noise_room, think_trace=think, category="undo_multi_action"))

    return examples

def gen_pronoun_them_door_scope(target: int = 1_500) -> list:
    examples = []
    open_p  = ["Open them.", "Unlock them.", "Open those doors.", "Undo them."]
    close_p = ["Close them.", "Lock them.", "Lock those doors.", "Secure them."]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_doors=5)
        state = generate_random_state(avail_r, avail_d)
        n_log     = random.randint(2, min(4, len(avail_d) - 1))
        log_doors = random.sample(avail_d, n_log)
        log_s     = random.choice(["lock", "unlock"])
        log_aw    = "locked" if log_s == "lock" else "unlocked"
        for d in log_doors:
            apply_force(state, {"doors": {d: log_aw}}, avail_r, avail_d)
        call_strs   = [f"lock_door(door={d}, state={log_s})" for d in log_doors]
        summary     = " ".join(f"{DOOR_DISPLAY[d].title()} {log_aw}." for d in log_doors)
        action_log  = fmt_txn(random.randint(1, 4), call_strs, summary)
        
        rev_s  = "unlock" if log_s == "lock" else "lock"
        rev_aw = "unlocked" if log_s == "lock" else "locked"
        prompt = random.choice(open_p if rev_s == "unlock" else close_p)
        calls  = [{"name": "lock_door", "args": {"door": d, "state": rev_s}} for d in log_doors]
        resp   = " ".join(f"{DOOR_DISPLAY[d].title()} {rev_aw}." for d in log_doors)
        door_names = ", ".join(log_doors)
        
        think = (
            f"User said '{prompt}'. "
            f"Pronoun 'them'/'those' resolves strictly to the {n_log} devices in the first [...] block. "
            f"Logged doors: {door_names}. "
            f"Issuing exactly {n_log} lock_door(state={rev_s}) calls."
        )
        noise_room = random.choice([""] + avail_r)
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=noise_room, action_log=action_log, think_trace=think, category="pronoun_them_door_scope"))
    return examples

def gen_social_off_topic(target: int = 800) -> list:
    examples = []
    SOCIAL = [
        ("Thank you.",    "You're welcome!"),
        ("Thanks!",       "No problem!"),
        ("thank you",     "Happy to help!"),
        ("Great.",        "Glad that worked!"),
        ("Perfect.",      "Great!"),
        ("Hello.",        "Hi there! What would you like to do?"),
        ("Hi",            "Hello! How can I help?"),
        ("Ok.",           "Got it! Let me know if you need anything."),
        ("Okay",          "Sure, let me know if there's anything else."),
    ]
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        prompt, reply = random.choice(SOCIAL)
        action_log = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2))
        noise_room = random.choice(["", random.choice(avail_r)])
        think = (
            f"User said '{prompt}'. "
            f"This is a social/conversational phrase — not a device control command. "
            f"Responding with a friendly reply. No tool call."
        )
        examples.append(build_ex(prompt, [], reply, avail_r, avail_d, state,
            user_room=noise_room, action_log=action_log, think_trace=think, category="social_off_topic"))
    return examples

def gen_implicit_light_user_room_strict(target: int = 1_500) -> list:
    examples = []
    on_p  = ["On the light.", "Turn the light on.", "On this light.", "Switch on the light.", "Switch the light on"]
    off_p = ["Off the light.", "Turn the light off.", "Off this light.", "Switch off the light", "Switch the light off."]
    for _ in range(target):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        ls  = random.choice(["on", "off"])
        opp = "off" if ls == "on" else "on"
        apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)
        alias = random.choice(ROOM_ALIASES[r])
        dist_mins  = random.randint(1, 3)
        other_log  = get_cross_room_txn(avail_r, avail_d, r, dist_mins)
        action_log = other_log + "\n" + build_distractor_log(
            avail_r, avail_d, n=1, start_mins=dist_mins + random.randint(5, 10))
        prompt = random.choice(on_p if ls == "on" else off_p)
        think  = (
            f"User is in '{r}'. User said '{prompt}'. "
            f"Generic 'the light'/'this light' resolves to current_user_room='{r}'. "
            f"Checking STATE: {r}={opp}. Mismatch — user wants {ls}. "
            f"Calling toggle_lights(room={r}, state={ls})."
        )
        examples.append(build_ex(prompt,
            [{"name": "toggle_lights", "args": {"room": r, "state": ls}}],
            f"The {alias} light is now {ls}.", avail_r, avail_d, state,
            user_room=r, action_log=action_log, think_trace=think, category="implicit_light_user_room_strict"))
    return examples
    
def gen_this_vs_that_door(target: int = 1_000) -> list:
    examples = []
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_doors=4)
        state = generate_random_state(avail_r, avail_d)
        log_d = random.choice(avail_d)
        log_s = random.choice(["lock", "unlock"])
        log_aw = "locked" if log_s == "lock" else "unlocked"
        primary_mins = random.randint(1, 4)
        action_log = fmt_txn(primary_mins, [f"lock_door(door={log_d}, state={log_s})"], f"{DOOR_DISPLAY[log_d]} {log_aw}.")
        candidates = [d for d in ROOM_DOOR_ROOMS if d in avail_d and d != log_d]
        if not candidates: continue
        u_room = random.choice(candidates)
        action = random.choice(["lock", "unlock"])
        aw_tgt = "locked" if action == "lock" else "unlocked"

        if random.random() < 0.5:
            apply_force(state, {"doors": {u_room: "unlocked" if action == "lock" else "locked"}}, avail_r, avail_d)
            prompt = random.choice([f"{action.capitalize()} this door.", f"{'Close' if action=='lock' else 'Open'} this door."])
            calls = [{"name": "lock_door", "args": {"door": u_room, "state": action}}]
            resp  = f"The {DOOR_DISPLAY[u_room]} is now {aw_tgt}."
            think = (
                f"User said '{prompt}'. "
                f"'this door' resolves to current_user_room='{u_room}'. "
                f"Calling lock_door(door={u_room}, state={action})."
            )
        else:
            apply_force(state, {"doors": {log_d: "unlocked" if action == "lock" else "locked"}}, avail_r, avail_d)
            t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
            prompt = random.choice([f"{action.capitalize()} that door.", f"{'Close' if action=='lock' else 'Open'} that one.", "Undo that door."])
            calls = [{"name": "lock_door", "args": {"door": log_d, "state": action}}]
            resp  = f"The {DOOR_DISPLAY[log_d]} is now {aw_tgt}."
            think = (
                f"User said '{prompt}'. "
                f"'that door' / 'that one' resolves to the first [...] block "
                f"in RECENT ACTIONS ({t_label}): {DOOR_DISPLAY[log_d]}. "
                f"Calling lock_door(door={log_d}, state={action})."
            )
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=u_room, action_log=action_log, think_trace=think, category="this_vs_that_door"))
    return examples


def gen_pronoun_crossroom(target: int = 900) -> list:
    """GAP F: pronoun resolves to log device, not current room."""
    examples = []
    off_p = ["Off it.", "Turn it off.", "Kill it.", "Shut it off.",
              "Switch it off.", "Off it now.", "Can you off it?"]

    on_p  = ["On it.", "Turn it on.", "Switch it on.", "Put it back on.",
              "On it back.", "Turn it back on.", "Bring it back on.",
              "On it please.", "Put it back.", "Switch it back on."]

    for _ in range(target):
        r_log  = random.choice(ALL_ROOMS)
        r_user = random.choice([x for x in ALL_ROOMS if x != r_log])
        avail_r, avail_d = sample_topology(required_rooms=[r_log, r_user])
        state = generate_random_state(avail_r, avail_d)
        s_log    = random.choice(["on", "off"])
        target_s = "off" if s_log == "on" else "on"
        apply_force(state, {"lights": {r_log: s_log}}, avail_r, avail_d)
        primary_mins = random.randint(1, 4)
        primary_txn  = fmt_txn(primary_mins,
                               [f"toggle_lights(room={r_log}, state={s_log})"],
                               f"{ROOM_DISPLAY[r_log]} light turned {s_log}.")
        if random.random() < 0.6:
            dist = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2),
                                        start_mins=primary_mins + random.randint(7, 14))
            action_log = primary_txn + "\n" + dist
        else:
            action_log = primary_txn
        alias_log = random.choice(ROOM_ALIASES[r_log])
        prompt    = random.choice(off_p if target_s == "off" else on_p)
        t_label   = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        think = (
            f"User said '{prompt}'. "
            f"Pronoun 'it' → first [...] block ({t_label}): {ROOM_DISPLAY[r_log]} light. "
            f"STATE shows {r_log}:{s_log}. User wants '{target_s}' (opposite). "
            f"Calling toggle_lights(room={r_log}, state={target_s})."
        )

        examples.append(build_ex(prompt,
            [{"name": "toggle_lights", "args": {"room": r_log, "state": target_s}}],
            f"The {alias_log} light is now {target_s}.",
            avail_r, avail_d, state,
            user_room=r_user, action_log=action_log,
            think_trace=think, category="pronoun_crossroom"))
    return examples


def gen_speaker_explicit_stop(target: int = 800) -> list:
    """
    GAP G: explicit-room stop → action=stop.
    FIX-B: Positive-only think trace.
    """
    examples = []
    stop_tmpls = [
        "Stop the {r} speaker.",     "Off the {r} speaker.",
        "Turn off the {r} speaker.", "Off the music in the {r}.",
        "Stop the music in the {r}.","Kill the {r} speaker.",
        "Stop the {r} music.",       "Silence the {r} speaker.",
    ]
    for _ in range(target):
        sp_rooms_avail = random.sample(
            [r for r in ALL_ROOMS if r in SPEAKER_ROOMS],
            random.randint(1, min(3, len(SPEAKER_ROOMS))))
        avail_r, avail_d = sample_topology(required_rooms=sp_rooms_avail)
        state = generate_random_state(avail_r, avail_d)
        r = random.choice(sp_rooms_avail)
        state["speaker"][r] = "playing"
        alias  = random.choice(ROOM_ALIASES[r])
        prompt = random.choice(stop_tmpls).format(r=alias)
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""
        # NOISE: explicit room name wins regardless of where user is
        noise_room = random.choice(["", r, random.choice(avail_r)])
        loc_str = f"current_user_room='{noise_room}'" if noise_room else "user room is unknown"
        think = (
            f"User wants to stop the {alias} speaker. "
            f"Target room is '{r}'. "
            f"'Off/stop the speaker' maps to action='stop'. "
            f"Speaker is currently playing. "
            f"Calling control_speaker(room={r}, action=stop)."
        )
        examples.append(build_ex(prompt,
            [{"name": "control_speaker", "args": {"room": r, "action": "stop"}}],
            f"Stopped the music on the {alias} speaker.",
            avail_r, avail_d, state, action_log=action_log,
            user_room=noise_room, think_trace=think, category="speaker_explicit_stop"))
    return examples


def gen_speaker_resume_synonyms(target: int = 600) -> list:
    """GAP H: continue/resume/on it back → action=play."""
    examples = []
    resume_p = [
        "Continue the music.", "Resume the music.", "Resume.", "Continue.",
        "On it back.", "Play it again.", "Unpause.", "Un-pause the music.",
        "Start the music again.", "Continue playing.", "Keep playing.",
        "Continue the song.", "Play it.", "On the music.", "On the speaker.",
    ]
    for _ in range(target):
        sp_rooms_avail = [r for r in ALL_ROOMS if r in SPEAKER_ROOMS]
        r = random.choice(sp_rooms_avail)
        avail_r, avail_d = sample_topology(required_rooms=[r])

        # FIX: Prune extra speakers so the think trace is truthful
        avail_r = [x for x in avail_r if x not in SPEAKER_ROOMS or x == r]
        state = generate_random_state(avail_r, avail_d)
        state["speaker"][r] = "paused"
        alias     = random.choice(ROOM_ALIASES[r])
        user_room = r if random.random() < 0.5 else ""
        prompt    = random.choice(resume_p)
        spk_list = [x for x in avail_r if x in SPEAKER_ROOMS]
        conn_str = ", ".join(spk_list)

        think     = (
            f"User said '{prompt}'. "
            f"'Continue'/'resume'/'on the music'/'on it back' all map to action='play'. "
            f"Checking CONNECTED SPEAKERS: [{conn_str}]. Exactly one speaker connected. "
            f"Speaker is currently paused. "
            f"Calling control_speaker(room={r}, action=play)."
        )
        examples.append(build_ex(prompt,
            [{"name": "control_speaker", "args": {"room": r, "action": "play"}}],
            f"Resuming music on the {alias} speaker.",
            avail_r, avail_d, state, user_room=user_room,
            think_trace=think, category="speaker_resume_synonyms"))
    return examples


def gen_thermostat_reinforced(target: int = 1_200) -> list:
    """GAP J: set_thermostat must always be called when value differs."""
    examples = []
    tmpls = [
        "Set temp to {v}.",        "Set it to {v} degrees.",
        "Thermostat to {v}.",      "Set the temperature to {v}.",
        "Make it {v} degrees.",    "Change temp to {v}.",
        "I want it at {v}.",       "{v} degrees please.",
        "Turn the thermostat to {v}.", "Set temp to {v} please.",
    ]
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        cur   = state["thermostat"]["temperature"]
        val   = random.randint(MIN_T, MAX_T)
        while val == cur: val = random.randint(MIN_T, MAX_T)
        mode   = "cool" if val < cur else "heat"
        prompt = random.choice(tmpls).format(v=val)
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""
        u_room = random.choice(["", random.choice(avail_r)])
        loc_str = f"current_user_room='{u_room}' is irrelevant" if u_room else "user room is unknown and irrelevant"
        
        # UPDATE THINK TRACES:
        think = (
            f"User wants thermostat at {val}F. Current: {cur}F/{state['thermostat']['mode']}. "
            f"Values differ — tool call required. "
            f"Calling set_thermostat(temperature={val}, mode='{mode}')."
        )
        examples.append(build_ex(prompt,
            [{"name": "set_thermostat", "args": {"temperature": val, "mode": mode}}],
            f"Thermostat set to {val}°F in {mode} mode.",
            avail_r, avail_d, state, action_log=action_log,
            think_trace=think,user_room=u_room, category="thermostat_reinforced"))
    return examples


def gen_compound_three_action(target: int = 2_000) -> list:
    """GAP L: 3-action compound — all three must execute."""
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3)
        state = generate_random_state(avail_r, avail_d)
        combo = random.choice(["scene_light_door", "light_light_scene",
                                "light_tv_scene", "all_off_tv_scene","light_light_door"])

        if combo == "scene_light_door":
            scene = random.choice(SCENES)
            r = random.choice(avail_r); d = random.choice(avail_d)
            ls = random.choice(["on", "off"]); ds = random.choice(["lock", "unlock"])
            ra = random.choice(ROOM_ALIASES[r]); da = random.choice(DOOR_ALIASES[d])
            aw = "locked" if ds == "lock" else "unlocked"
            apply_force(state, {"lights": {r: ("off" if ls=="on" else "on")}}, avail_r, avail_d)
            apply_force(state, {"doors":  {d: ("unlocked" if ds=="lock" else "locked")}}, avail_r, avail_d)
            prompts = [
                f"Set scene to {scene.replace('_',' ')} and {ls} the {ra} light and {ds} the {da}.",
                f"Activate {scene.replace('_',' ')} mode, {ls} the {ra} light, and {ds} the {da}.",
            ]
            calls = [{"name": "set_scene",    "args": {"scene": scene}},
                     {"name": "toggle_lights", "args": {"room": r, "state": ls}},
                     {"name": "lock_door",     "args": {"door": d, "state": ds}}]
            think = (f"Compound request. Counting sub-actions: (1) set_scene(scene={scene}), "
                     f"(2) toggle_lights(room={r}, state={ls}), "
                     f"(3) lock_door(door={d}, state={ds}).")
            resp = f"{SCENE_RESP[scene]} {ra.title()} light {ls}. {da.title()} {aw}."

        elif combo == "light_light_scene":
            r1, r2 = random.sample(avail_r, 2)
            scene  = random.choice(SCENES)
            ls1 = random.choice(["on","off"]); ls2 = random.choice(["on","off"])
            ra1 = random.choice(ROOM_ALIASES[r1]); ra2 = random.choice(ROOM_ALIASES[r2])
            apply_force(state, {"lights": {r1: ("off" if ls1=="on" else "on"),
                                           r2: ("off" if ls2=="on" else "on")}}, avail_r, avail_d)
            prompts = [
                f"{ls1.capitalize()} the {ra1} light, {ls2} the {ra2} light, and set {scene.replace('_',' ')} mode.",
            ]
            calls = [{"name": "toggle_lights", "args": {"room": r1, "state": ls1}},
                     {"name": "toggle_lights", "args": {"room": r2, "state": ls2}},
                     {"name": "set_scene",     "args": {"scene": scene}}]
            think = (f"Compound request. Counting sub-actions: (1) toggle_lights(room={r1}, state={ls1}), "
                     f"(2) toggle_lights(room={r2}, state={ls2}), "
                     f"(3) set_scene(scene={scene}).")
            resp = (f"{ra1.title()} light {ls1}. {ra2.title()} light {ls2}. "
                    f"{SCENE_RESP[scene]}")

        elif combo == "light_tv_scene":
            tv_rooms = [x for x in avail_r if x in TV_ROOMS]
            if not tv_rooms: continue
            tr = tv_rooms[0]; r = random.choice([x for x in avail_r if x != tr])
            scene = random.choice(SCENES)
            ls = random.choice(["on","off"]); ts = random.choice(["on","off"])
            ra = random.choice(ROOM_ALIASES[r]); ta = random.choice(ROOM_ALIASES[tr])
            apply_force(state, {"lights": {r: ("off" if ls=="on" else "on")}}, avail_r, avail_d)
            state["tv"][tr] = "off" if ts == "on" else "on"
            prompts = [
                f"Set {scene.replace('_',' ')} mode, turn {ls} the {ra} light, and {ts} the {ta} TV.",
            ]
            calls = [{"name": "set_scene",    "args": {"scene": scene}},
                     {"name": "toggle_lights", "args": {"room": r, "state": ls}},
                     {"name": "control_tv",    "args": {"room": tr, "state": ts}}]
            think = (f"Compound request. Counting sub-actions: (1) set_scene(scene={scene}), "
                     f"(2) toggle_lights(room={r}, state={ls}), "
                     f"(3) control_tv(room={tr}, state={ts}).")
            resp = f"{SCENE_RESP[scene]} {ra.title()} light {ls}. {ta.title()} TV {ts}."

        elif combo == "light_light_door":
            r1, r2 = random.sample(avail_r, 2)
            d = random.choice(avail_d)
            ls1 = random.choice(["on","off"]); ls2 = random.choice(["on","off"]); ds = random.choice(["lock","unlock"])
            ra1 = random.choice(ROOM_ALIASES[r1]); ra2 = random.choice(ROOM_ALIASES[r2]); da = random.choice(DOOR_ALIASES[d])
            apply_force(state, {"lights": {r1: "off" if ls1=="on" else "on", r2: "off" if ls2=="on" else "on"}, "doors": {d: "unlocked" if ds=="lock" else "locked"}}, avail_r, avail_d)
            
            prompts = [f"Turn {ls1} the {ra1} light, turn {ls2} the {ra2} light, and {ds} the {da}."]
            calls = [{"name": "toggle_lights", "args": {"room": r1, "state": ls1}},
                     {"name": "toggle_lights", "args": {"room": r2, "state": ls2}},
                     {"name": "lock_door", "args": {"door": d, "state": ds}}]
            think = (f"Compound request. Counting sub-actions: (1) toggle_lights(room={r1}, state={ls1}), "
                     f"(2) toggle_lights(room={r2}, state={ls2}), "
                     f"(3) lock_door(door={d}, state={ds}).")
            resp = f"{ra1.title()} light {ls1}. {ra2.title()} light {ls2}. {da.title()} {'locked' if ds=='lock' else 'unlocked'}."

        else:  # all_off_tv_scene
            tv_rooms = [x for x in avail_r if x in TV_ROOMS]
            if not tv_rooms: continue
            tr = tv_rooms[0]; scene = random.choice(SCENES)
            ts = random.choice(["on","off"]); ta = random.choice(ROOM_ALIASES[tr])
            
            on_rs = [x for x in avail_r if state["lights"][x]["state"] == "on"]
            l_summary = ", ".join(f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
            
            if not on_rs:
                # Lights all already off — sub-action 1 produces 0 calls
                light_calls = []
                light_resp = "All lights are already off."
                light_think = (
                    f"(1) 'Turn off all lights' — checking ALL connected lights: {l_summary}. "
                    f"Result: 0 lights currently on — all {len(avail_r)} lights already off. No toggle_lights calls needed."
                )
            else:
                light_calls = [{"name": "toggle_lights", "args": {"room": r, "state": "off"}} for r in on_rs]
                on_names = ", ".join(ROOM_DISPLAY[r] for r in on_rs)
                light_think = (
                    f"(1) 'Turn off all lights' — checking ALL connected lights: {l_summary}. "
                    f"Result: {len(on_rs)} light(s) on ({on_names}). "
                    f"Issuing {len(on_rs)} toggle_lights(off) call(s) individually."
                )
                light_resp = " ".join(f"{ROOM_DISPLAY[r].title()} light off." for r in on_rs)
            
            # Sub-actions 2 and 3 (TV and Scene) execute normally regardless of the lights
            state["tv"][tr] = "off" if ts == "on" else "on"
            
            calls = light_calls + [
                {"name": "control_tv", "args": {"room": tr, "state": ts}},
                {"name": "set_scene",  "args": {"scene": scene}},
            ]
            
            think = (
                f"Compound request. Counting sub-actions: "
                f"{light_think} "
                f"(2) control_tv(room={tr}, state={ts}). "
                f"(3) set_scene(scene={scene})."
            )
            
            resp = f"{light_resp} {ta.title()} TV {ts}. {SCENE_RESP[scene]}"
            
            prompts = [
                f"Turn off all lights, turn {ts} the {ta} TV, and activate {scene.replace('_', ' ')}.",
                f"Off all the lights, {ts} the {ta} TV, and set {scene.replace('_', ' ')} mode.",
            ]

        action_log = build_distractor_log(avail_r, avail_d, n=1) if random.random() < 0.3 else ""
        
        # build_ex automatically adds "Total: X tool calls required. Emitting all X. ACTION REQUIRED." 
        # so we don't manually append it here, avoiding double-count bugs.
        examples.append(build_ex(random.choice(prompts), calls, resp,
            avail_r, avail_d, state, action_log=action_log,
            think_trace=think, category="compound_three_action"))
    return examples


def gen_heterogeneous_undo(target: int = 1_000) -> list:
    """Undo a transaction containing mixed device types."""
    examples = []
    undo_phrases = ["Undo all that", "Revert that", "Take that last command back",
                    "Undo that"]
    for _ in range(target):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        actions_to_do = random.sample(["light", "door", "scene", "thermostat"],
                                      random.randint(2, 3))
        call_strs  = []
        calls      = []
        resp_parts = []
        think_parts = ["User said undo. Looking at first [...] block."]
        t_old = t_new = None
        primary_mins = random.randint(1, 4)

        for act in actions_to_do:
            if act == "light":
                r = random.choice(avail_r); s = random.choice(["on", "off"])
                opp = "off" if s == "on" else "on"
                apply_force(state, {"lights": {r: s}}, avail_r, avail_d)
                call_strs.append(f"toggle_lights(room={r}, state={s})")
                calls.append({"name": "toggle_lights", "args": {"room": r, "state": opp}})
                resp_parts.append(f"{ROOM_DISPLAY[r].title()} light turned {opp}.")
                think_parts.append(f"Light {r} was {s} → reversing to {opp}.")
            elif act == "door":
                d = random.choice(avail_d); s = random.choice(["lock", "unlock"])
                opp = "unlock" if s == "lock" else "lock"
                aw  = "locked" if s == "lock" else "unlocked"
                taw = "unlocked" if opp == "unlock" else "locked"
                apply_force(state, {"doors": {d: aw}}, avail_r, avail_d)
                call_strs.append(f"lock_door(door={d}, state={s})")
                calls.append({"name": "lock_door", "args": {"door": d, "state": opp}})
                resp_parts.append(f"{DOOR_DISPLAY[d].title()} {taw}.")
                think_parts.append(f"Door {d} was {s} → reversing to {opp}.")
            elif act == "scene":
                sc = random.choice(SCENES)
                call_strs.append(f"set_scene(scene={sc})")
                resp_parts.append(
                    f"I can't undo the {sc.replace('_', ' ').title()} scene directly.")
                think_parts.append(f"Scene {sc} — cannot be undone with a tool. Skipping.")
            elif act == "thermostat":
                t_new = random.randint(70, 75)
                t_old = random.randint(65, 69)
                m = "auto"
                apply_force(state, {"thermostat": {"temperature": t_new, "mode": m}},
                            avail_r, avail_d)
                call_strs.append(f"set_thermostat(temperature={t_new}, mode={m})")
                calls.append({"name": "set_thermostat",
                               "args": {"temperature": t_old, "mode": m}})
                resp_parts.append(f"Thermostat set back to {t_old}°F.")
                think_parts.append(f"Thermostat {t_old}→{t_new} → reversing to {t_old}.")

        summary = "Multiple actions executed."
        if "thermostat" in actions_to_do and t_old is not None:
            summary += f" (Thermostat was {t_old}F.)"

        primary_txn = fmt_txn(primary_mins, call_strs, summary)
        action_log  = primary_txn + "\n" + build_distractor_log(
            avail_r, avail_d, n=1, start_mins=primary_mins + 10)

        # NOISE — undo resolves via action log, not user location
        noise_room = random.choice(["", random.choice(avail_r)])
        loc_str = f" (ignoring current_user_room='{noise_room}')" if noise_room else " (user room is unknown)"

        prompt     = random.choice(undo_phrases)
        think      = f"User said '{prompt}'. " + " ".join(think_parts)
        final_resp = " ".join(resp_parts)

        examples.append(build_ex(prompt, calls, final_resp, avail_r, avail_d, state,
            action_log=action_log, user_room=noise_room, think_trace=think,
            category="heterogeneous_undo"))
    return examples


def gen_compound_log_and_local(target: int = 1_500) -> list:
    """GAP M: Compound mixing Log-based undo with a Local action."""
    examples = []
    log_phrases = ["Undo that", "Revert that", "Take that back"]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)
        log_type   = random.choice(["light", "door"])
        local_type = "door" if log_type == "light" else "light"
        is_ambiguous = random.choice([True, False])

        calls       = []
        resp_parts  = []
        think_parts = []
        primary_mins = random.randint(1, 4)
        log_r = None
        log_d = None

        if log_type == "light":
            log_r   = random.choice(avail_r)
            s_log   = random.choice(["on", "off"])
            tgt_s   = "off" if s_log == "on" else "on"
            apply_force(state, {"lights": {log_r: s_log}}, avail_r, avail_d)
            primary_txn = fmt_txn(primary_mins,
                                  [f"toggle_lights(room={log_r}, state={s_log})"],
                                  f"{ROOM_DISPLAY[log_r]} light turned {s_log}.")
            calls.append({"name": "toggle_lights",
                          "args": {"room": log_r, "state": tgt_s}})
            resp_parts.append(
                f"{ROOM_DISPLAY[log_r].title()} light turned {tgt_s}")
            think_parts.append(
                f"Part 1: undo first block → {ROOM_DISPLAY[log_r]} light {s_log} → {tgt_s}.")
        else:
            log_d   = random.choice(avail_d)
            s_log   = random.choice(["lock", "unlock"])
            tgt_s   = "unlock" if s_log == "lock" else "lock"
            aw      = "locked" if s_log == "lock" else "unlocked"
            taw     = "unlocked" if tgt_s == "unlock" else "locked"
            apply_force(state, {"doors": {log_d: aw}}, avail_r, avail_d)
            primary_txn = fmt_txn(primary_mins,
                                  [f"lock_door(door={log_d}, state={s_log})"],
                                  f"{DOOR_DISPLAY[log_d]} {aw}.")
            calls.append({"name": "lock_door",
                          "args": {"door": log_d, "state": tgt_s}})
            resp_parts.append(f"{DOOR_DISPLAY[log_d].title()} {taw}")
            think_parts.append(
                f"Part 1: undo first block → {DOOR_DISPLAY[log_d]} {aw} → {tgt_s}.")

        action_log = primary_txn + "\n" + build_distractor_log(
            avail_r, avail_d, n=1, start_mins=primary_mins + 10)

        cur_room = ""
        local_a1_prompt = ""

        if local_type == "door":
            local_verb = random.choice(["open", "close", "lock", "unlock"])
            local_s    = "unlock" if local_verb in ["open", "unlock"] else "lock"
            local_aw   = "unlocked" if local_s == "unlock" else "locked"
            local_a1_prompt = f"{local_verb} the door"
            if not is_ambiguous:
                candidates = [d for d in avail_d if d != log_d] if log_d else list(avail_d)
                if not candidates: candidates = list(avail_d)
                cur_room = random.choice(candidates)
                opp_aw   = "unlocked" if local_s == "lock" else "locked"
                apply_force(state, {"doors": {cur_room: opp_aw}}, avail_r, avail_d)
                calls.append({"name": "lock_door",
                              "args": {"door": cur_room, "state": local_s}})
                resp_parts.append(f"{DOOR_DISPLAY.get(cur_room, cur_room).title()} {local_aw}")
                think_parts.append(
                    f"Part 2: local door. current_user_room='{cur_room}'. "
                    f"Calling lock_door(door={cur_room}, state={local_s}).")
            else:
                calls.append({"name": "intent_unclear", "args": {"reason": "incomplete"}})
                think_parts.append(
                    "Part 2: local door. current_user_room empty. "
                    "Calling intent_unclear(incomplete).")
        else:
            local_verb  = random.choice(["turn on", "turn off", "on", "off"])
            local_s     = "on" if local_verb in ["turn on", "on"] else "off"
            local_a1_prompt = f"{local_verb} the light"
            if not is_ambiguous:
                candidates = [r for r in avail_r if r != log_r] if log_r else list(avail_r)
                if not candidates: candidates = list(avail_r)
                cur_room = random.choice(candidates)
                opp_s    = "off" if local_s == "on" else "on"
                apply_force(state, {"lights": {cur_room: opp_s}}, avail_r, avail_d)
                calls.append({"name": "toggle_lights",
                              "args": {"room": cur_room, "state": local_s}})
                resp_parts.append(
                    f"{ROOM_DISPLAY[cur_room].title()} light turned {local_s}")
                think_parts.append(
                    f"Part 2: local light. current_user_room='{cur_room}'. "
                    f"Calling toggle_lights(room={cur_room}, state={local_s}).")
            else:
                calls.append({"name": "intent_unclear", "args": {"reason": "incomplete"}})
                think_parts.append(
                    "Part 2: local light. current_user_room empty. "
                    "Calling intent_unclear(incomplete).")

        prompt = f"{random.choice(log_phrases)} and {local_a1_prompt}."
        think  = "Compound request. " + " ".join(think_parts)

        if is_ambiguous:
            clarify_t = "door" if local_type == "door" else "room's light"
            final_resp = (resp_parts[0].capitalize()
                          + f". However, which {clarify_t} did you mean?")
        else:
            final_resp = " and ".join(resp_parts) + "."
            final_resp = final_resp[0].upper() + final_resp[1:]

        examples.append(build_ex(prompt, calls, final_resp,
            avail_r, avail_d, state,
            user_room=cur_room, action_log=action_log,
            think_trace=think, category="compound_log_and_local"))
    return examples


# ── Clean response helpers ─────────────────────────────────────────────

def gen_clean_response(target: int = 800) -> list:
    examples = []
    attempts  = 0
    fns = [_clean_single_light, _clean_single_door, _clean_single_tv,
           _clean_single_scene, _clean_single_thermostat,
           _clean_multi_lights, _clean_multi_light_door]
    while len(examples) < target and attempts < target * 5:
        attempts += 1
        avail_r, avail_d = sample_topology()
        state  = generate_random_state(avail_r, avail_d)
        result = random.choice(fns)(avail_r, avail_d, state)
        if result: examples.append(result)
    return examples[:target]

def _clean_single_light(avail_r, avail_d, state):
    r = random.choice(avail_r); alias = random.choice(ROOM_ALIASES[r])
    s = random.choice(["on", "off"])
    apply_force(state, {"lights": {r: ("off" if s=="on" else "on")}}, avail_r, avail_d)
    think = (f"Turning {s} the {alias} light. "
             f"After <|tool_call_end|> output clean natural-language text only.")
    return build_ex(f"Turn {s} the {alias} light.",
        [{"name": "toggle_lights", "args": {"room": r, "state": s}}],
        f"The {alias} light is now {s}.", avail_r, avail_d, state,
        think_trace=think, category="clean_response")

def _clean_single_door(avail_r, avail_d, state):
    d = random.choice(avail_d); alias = random.choice(DOOR_ALIASES[d])
    s = random.choice(["lock", "unlock"]); aw = "locked" if s=="lock" else "unlocked"
    apply_force(state, {"doors": {d: ("unlocked" if s=="lock" else "locked")}}, avail_r, avail_d)
    think = f"{'Locking' if s=='lock' else 'Unlocking'} {alias}. Response after tag = plain English only."
    return build_ex(f"{'Lock' if s=='lock' else 'Unlock'} the {alias}.",
        [{"name": "lock_door", "args": {"door": d, "state": s}}],
        f"The {alias} is now {aw}.", avail_r, avail_d, state,
        think_trace=think, category="clean_response")

def _clean_single_tv(avail_r, avail_d, state):
    tv_rooms = [r for r in avail_r if r in TV_ROOMS]
    if not tv_rooms: return None
    r = random.choice(tv_rooms); alias = random.choice(ROOM_ALIASES[r])
    s = random.choice(["on", "off"])
    state["tv"][r] = "off" if s=="on" else "on"
    think = f"Turning {s} the {alias} TV. Clean English after tag."
    return build_ex(f"Turn {s} the {alias} TV.",
        [{"name": "control_tv", "args": {"room": r, "state": s}}],
        f"The {alias} TV is now {s}.", avail_r, avail_d, state,
        think_trace=think, category="clean_response")

def _clean_single_scene(avail_r, avail_d, state):
    scene = random.choice(SCENES)
    think = f"Setting scene '{scene}'. After tag: '{SCENE_RESP[scene]}' ."
    return build_ex(f"Set scene to {scene.replace('_',' ')}.",
        [{"name": "set_scene", "args": {"scene": scene}}],
        SCENE_RESP[scene], avail_r, avail_d, state,
        think_trace=think, category="clean_response")

def _clean_single_thermostat(avail_r, avail_d, state):
    cur = state["thermostat"]["temperature"]
    val = random.randint(MIN_T, MAX_T)
    while val == cur: val = random.randint(MIN_T, MAX_T)   # guarantee a change
    mode = "cool" if val < cur else "heat"
    think = f"Setting thermostat {val}F/{mode}. Response = plain English only."
    return build_ex(f"Set the thermostat to {val}.",
        [{"name": "set_thermostat", "args": {"temperature": val, "mode": mode}}],
        f"Thermostat set to {val}°F in {mode} mode.", avail_r, avail_d, state,
        think_trace=think, category="clean_response")

def _clean_multi_lights(avail_r, avail_d, state):
    if len(avail_r) < 2: return None
    rooms = random.sample(avail_r, random.randint(2, min(3, len(avail_r))))
    s = random.choice(["on", "off"])
    for r in rooms:
        apply_force(state, {"lights": {r: ("off" if s=="on" else "on")}}, avail_r, avail_d)
    aliases = [random.choice(ROOM_ALIASES[r]) for r in rooms]
    calls   = [{"name": "toggle_lights", "args": {"room": r, "state": s}} for r in rooms]
    resp    = " ".join(f"{a.title()} light {s}." for a in aliases)
    think   = ("Multiple tool calls each in <|tool_call_start|>...<|tool_call_end|>. "
               "After all calls, response is clean English only.")
    return build_ex(f"Turn {s} the {' and '.join(aliases)} lights.", calls, resp,
        avail_r, avail_d, state, think_trace=think, category="clean_response")

def _clean_multi_light_door(avail_r, avail_d, state):
    r = random.choice(avail_r); d = random.choice(avail_d)
    ls = random.choice(["on","off"]); ds = random.choice(["lock","unlock"])
    apply_force(state, {"lights": {r: ("off" if ls=="on" else "on")}}, avail_r, avail_d)
    apply_force(state, {"doors":  {d: ("unlocked" if ds=="lock" else "locked")}}, avail_r, avail_d)
    ra = random.choice(ROOM_ALIASES[r]); da = random.choice(DOOR_ALIASES[d])
    calls = [{"name": "toggle_lights", "args": {"room": r, "state": ls}},
             {"name": "lock_door",     "args": {"door": d, "state": ds}}]
    resp  = f"{ra.title()} light {ls} and {da} {'locked' if ds=='lock' else 'unlocked'}."
    think = "Two tool calls. Final response is pure natural language ."
    return build_ex(f"Turn {ls} the {ra} light and {ds} the {da}.", calls, resp,
        avail_r, avail_d, state, think_trace=think, category="clean_response")


# ── Gadgets (TV, Speaker, Fan) ─────────────────────────────────────────

TV_ON_TMPLS  = ["Turn on the TV.", "TV on.", "On the TV.", "Start the TV."]
TV_OFF_TMPLS = ["Turn off the TV.", "TV off.", "Off the TV.", "Power off the TV."]
TV_ON_EXPLICIT  = ["Turn on the {r} TV.", "Switch on the {r} TV.", "On the {r} TV."]
TV_OFF_EXPLICIT = ["Turn off the {r} TV.", "Switch off the {r} TV.", "Off the {r} TV."]

SPEAKER_PLAY_TMPLS = ["Play music.", "Play me some music.", "On the speaker.",
                       "On the music.", "Start the music.", "Play it."]
SPEAKER_PAUSE_TMPLS = ["Pause.", "Pause the music.", "Pause it.", "Hold the music."]
SPEAKER_STOP_TMPLS  = ["Stop the music.", "Off the speaker.", "Off the music.",
                        "Stop the speaker.", "Stop it.", "Kill the music.",
                        "Turn off the music.", "Cut the music."]
SPEAKER_NEXT_TMPLS  = ["Next.", "Next music.", "Play next music.", "Next song.",
                        "Skip.", "Next please.", "Skip the music."]
SPEAKER_PREV_TMPLS  = ["Previous.", "Previous song.", "Go back.", "Back.",
                        "Previous music.", "Play that last song again."]

SPEAKER_PLAY_EXPLICIT  = ["Play music in the {r}.", "Start the {r} speaker.",
                           "On the {r} speaker."]
SPEAKER_PAUSE_EXPLICIT = ["Pause the {r} speaker.", "Pause the music in the {r}."]
SPEAKER_STOP_EXPLICIT  = ["Stop the music in the {r}.", "Turn off the {r} speaker.",
                           "Off the {r} speaker.", "Stop the {r} music."]
SPEAKER_NEXT_EXPLICIT  = ["Next song in the {r}.", "Skip in the {r}."]
SPEAKER_PREV_EXPLICIT  = ["Previous song in the {r}.", "Go back in the {r}."]

FAN_ON_TMPLS  = ["Turn on the fan.", "Fan on.", "On the fan.", "Switch on the fan."]
FAN_OFF_TMPLS = ["Turn off the fan.", "Fan off.", "Off the fan.", "Switch off the fan."]
FAN_SPEED_TMPLS = ["Set the {r} fan to {sp} speed.", "Put the {r} fan on {sp}."]
FAN_ON_EXPLICIT  = ["Turn on the {r} fan.", "Switch on the {r} fan.", "On the {r} fan."]
FAN_OFF_EXPLICIT = ["Turn off the {r} fan.", "Switch off the {r} fan.", "Off the {r} fan."]


def _resolve_gadget(
    avail_r, avail_d, state, device_type, connected_rooms, desired_action,
    implicit_tmpls, explicit_tmpls_on, explicit_tmpls_off, already_sat_check,
    build_call_fn, resp_fn, clarify_resp_fn, category, scenario_weights=None,
    action_log="", eligible_state_for_infer=None, media=None
):
    if not connected_rooms: return None
    
    def get_tool_str(dt, rm, act, med=None):
        if dt == "tv": return f"control_tv(room={rm}, state={act})"
        if dt == "speaker":
            # Fix 4: Use double quotes to avoid breaking on songs like "It's Time"
            m_part = f', media="{med}"' if (act == "play" and med) else ""
            return f"control_speaker(room={rm}, action={act}{m_part})"
        return f"control_fan(room={rm}, state={act})"

    def get_actual_call(rm, act):
        c = build_call_fn(rm, act)
        if device_type == "speaker" and act == "play" and media:
            c["args"]["media"] = media
        return c

    # Fix 2: Proper conjugation for the "already satisfied" text replies
    sat_str = desired_action
    if device_type == "speaker":
        if desired_action == "play": sat_str = "playing"
        elif desired_action == "stop": sat_str = "stopped"
        elif desired_action == "pause": sat_str = "paused"

    weights = scenario_weights or [20, 20, 20, 20, 20]
    
    if len(connected_rooms) == 1:
        scenario = random.choices(["A", "B"], weights=[30, 70])[0]
    else:
        if len(weights) == 5:
            w = [weights[0], weights[2], weights[3], weights[4]]
        else:
            w = [25, 25, 25, 25]
        scenario = random.choices(["A", "C", "D", "E"], weights=w)[0]

    if scenario == "A":
        r = random.choice(connected_rooms); alias = random.choice(ROOM_ALIASES[r])
        noise_room = random.choice(["", r, random.choice(avail_r)]) 
        tmpls = explicit_tmpls_on if "on" in desired_action or \
            desired_action in ("play", "resume") else explicit_tmpls_off
        tmpl  = random.choice(tmpls).format(r=alias, r_t=alias.title())
        
        base_think = (f"User explicitly requested '{desired_action}' for the {alias} {device_type}. "
                      f"Target room is '{r}'. ")
                 
        # Fix 3: Force tool call if media is requested, even if already playing
        if already_sat_check(state, r) and not media:
            think = base_think + f"STATE shows {r} is already {sat_str}. No tool call needed."
            return build_ex(tmpl, [], f"The {alias} {device_type} is already {sat_str}.",
                avail_r, avail_d, state, user_room=noise_room, action_log=action_log,
                think_trace=think, category=category)
                
        tool_call_str = get_tool_str(device_type, r, desired_action, media)
        think = base_think + f"Calling {tool_call_str}."
        return build_ex(tmpl, [get_actual_call(r, desired_action)],
            resp_fn(alias, desired_action), avail_r, avail_d, state,
            user_room=noise_room, action_log=action_log, think_trace=think, category=category)

    elif scenario == "B":
        r = connected_rooms[0]; alias = random.choice(ROOM_ALIASES[r])
        noise_room = random.choice(["", r, random.choice(avail_r)]) 
        tmpl  = random.choice(implicit_tmpls)
        
        base_think = (f"Checking CONNECTED {device_type.upper()}S: [{r}]. "
                      f"Exactly one connected. Resolving to {r}. ")
                 
        if already_sat_check(state, r) and not media:
            think = base_think + f"STATE shows {r} is already {sat_str}. No tool call needed."
            return build_ex(tmpl, [], f"The {device_type} is already {sat_str}.",
                avail_r, avail_d, state, user_room=noise_room, action_log=action_log,
                think_trace=think, category=category)
                
        tool_call_str = get_tool_str(device_type, r, desired_action, media)
        think = base_think + f"Calling {tool_call_str}."
        return build_ex(tmpl, [get_actual_call(r, desired_action)],
            resp_fn(alias, desired_action), avail_r, avail_d, state,
            user_room=noise_room, action_log=action_log, think_trace=think, category=category)

    elif scenario == "C":
        r = random.choice(connected_rooms); alias = random.choice(ROOM_ALIASES[r])
        tmpl  = random.choice(implicit_tmpls)
        conn_str = ", ".join(connected_rooms)
        
        base_think = (f"Checking CONNECTED {device_type.upper()}S: [{conn_str}]. "
                      f"Multiple {device_type}s connected. "
                      f"current_user_room='{r}' has a {device_type}. "
                      f"Resolving to {r}. ")
                 
        if already_sat_check(state, r) and not media:
            think = base_think + f"STATE shows {r} is already {sat_str}. No tool call needed."
            return build_ex(tmpl, [], f"The {alias} {device_type} is already {sat_str}.",
                avail_r, avail_d, state, user_room=r, action_log=action_log,
                think_trace=think, category=category)
                
        tool_call_str = get_tool_str(device_type, r, desired_action, media)
        think = base_think + f"Calling {tool_call_str}."
        return build_ex(tmpl, [get_actual_call(r, desired_action)],
            resp_fn(alias, desired_action), avail_r, avail_d, state,
            user_room=r, action_log=action_log, think_trace=think, category=category)

    elif scenario == "D":
        if len(connected_rooms) < 2: return None
        r_target = connected_rooms[0]
        
        for r_other in connected_rooms[1:]:
            if device_type == "tv":
                state["tv"][r_other] = "on" if desired_action == "on" else "off"
            elif device_type == "speaker":
                infer_state = eligible_state_for_infer or ("stopped" if desired_action == "play" else "playing")
                other_state = "playing" if infer_state == "stopped" else "stopped"
                state["speaker"][r_other] = other_state
            else:
                state["fan"][r_other]["state"] = "on" if desired_action == "on" else "off"
                
        if device_type == "tv":
            state["tv"][r_target] = "off" if desired_action == "on" else "on"
        elif device_type == "speaker":
            infer_state = eligible_state_for_infer or ("stopped" if desired_action == "play" else "playing")
            state["speaker"][r_target] = infer_state
        else:
            state["fan"][r_target]["state"] = "off" if desired_action == "on" else "on"

        non_connected = [r for r in avail_r if r not in connected_rooms]
        if not non_connected:
            return None
        user_r = random.choice(non_connected)
        alias  = random.choice(ROOM_ALIASES[r_target])
        tmpl   = random.choice(implicit_tmpls)
        
        conn_str = ", ".join(connected_rooms)
        loc_context = f"User is in '{user_r}' which has no {device_type}." if user_r else "User's location is unknown."
        tool_call_str = get_tool_str(device_type, r_target, desired_action, media)
        think = (
            f"Checking CONNECTED {device_type.upper()}S: [{conn_str}]. "
            f"Multiple {device_type}s connected. "
            f"Checking states: exactly ONE {device_type} ({r_target}) is in the eligible state for '{desired_action}'. "
            f"Inferring {r_target}. "
            f"Calling {tool_call_str}."
        )
        return build_ex(tmpl, [get_actual_call(r_target, desired_action)],
            resp_fn(alias, desired_action), avail_r, avail_d, state,
            user_room=user_r, action_log=action_log, think_trace=think, category=category)

    else:  # E
        if len(connected_rooms) < 2: return None
        for r in connected_rooms:
            if device_type == "tv":
                state["tv"][r] = "on" if desired_action == "off" else "off"
            elif device_type == "speaker":
                state["speaker"][r] = "stopped" if desired_action == "play" else "playing"
            else:
                state["fan"][r]["state"] = "on" if desired_action == "off" else "off"
        user_r = random.choice([r for r in avail_r if r not in connected_rooms] or [""])
        tmpl   = random.choice(implicit_tmpls)
        
        conn_str = ", ".join(connected_rooms)
        loc_context = f"User is in '{user_r}' which has no {device_type}." if user_r else "User's location is unknown."
        think  = (f"Checking CONNECTED {device_type.upper()}S: [{conn_str}]. "
                  f"Multiple {device_type}s connected. "
                  f"All are in the eligible state. "
                  f"Cannot determine which one to act on. "
                  f"Calling intent_unclear(incomplete).")

        return build_ex(tmpl, [{"name": "intent_unclear", "args": {"reason": "incomplete"}}],
            clarify_resp_fn(connected_rooms), avail_r, avail_d, state,
            user_room=user_r, action_log=action_log, think_trace=think, category=category)

def gen_bulk_plus_gadget(target: int = 1_500) -> list:
    examples = []
 
    light_bulk = [
        "turn off the lights that are on", "off the lights that are on",
        "kill the lights that are on", "turn off what's on",
        "switch off the on lights"
    ]
    door_bulk = [
        "close the doors that are open", "lock the doors that are unlocked",
        "close any doors that are open", "secure the open doors"
    ]
    speaker_phrases = ["play some music", "put on some tunes", "start the music", "play music"]
    tv_phrases      = ["turn on the TV", "on the TV", "start the TV"]
    fan_phrases     = ["turn on the fan", "on the fan", "start the fan"]
 
    attempts = 0
    while len(examples) < target and attempts < target * 3:
        attempts += 1
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=3)
 
        bulk_type   = random.choice(["light", "door"])
        gadget_type = random.choice(["speaker", "tv", "fan"])
 
        # --- DEFINE ROOM LISTS BEFORE USING THEM ---
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        tv_rooms = [r for r in avail_r if r in TV_ROOMS]
        fan_rooms = [r for r in avail_r if r in FAN_ROOMS]

        target_r = ""
        if gadget_type == "speaker":
            if not sp_rooms: continue
            target_r = sp_rooms[0]
        elif gadget_type == "tv":
            if not tv_rooms: continue
            target_r = tv_rooms[0]
        else:
            if not fan_rooms: continue
            target_r = fan_rooms[0]
 
        # Generate state AFTER target_r is selected
        state = generate_random_state(avail_r, avail_d)
 
        calls = []
        resp_parts = []
 
        # ── Part 1: Bulk State-Aware Action ──
        if bulk_type == "light":
            n_on = random.randint(2, len(avail_r))
            on_rooms = random.sample(avail_r, n_on)
            for r in avail_r:
                state["lights"][r]["state"] = "on" if r in on_rooms else "off"
 
            on_names    = ", ".join(ROOM_DISPLAY[r] for r in on_rooms)
            all_summary = ", ".join(f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
 
            for r in on_rooms:
                calls.append({"name": "toggle_lights", "args": {"room": r, "state": "off"}})
 
            resp_parts.append(" ".join(f"{ROOM_DISPLAY[r].title()} light off." for r in on_rooms))
            part1_prompt = random.choice(light_bulk)
            part1_think = (
                f"(1) '{part1_prompt}': "
                f"Checking ALL connected lights: {all_summary}. "
                f"{len(on_rooms)} light(s) currently on ({on_names}). "
                f"Issuing {len(on_rooms)} toggle_lights(off) calls."
            )
        else:
            n_unl = random.randint(2, len(avail_d))
            unl_doors = random.sample(avail_d, n_unl)
            for d in avail_d:
                state["doors"][d] = "unlocked" if d in unl_doors else "locked"
 
            unl_names   = ", ".join(DOOR_DISPLAY[d] for d in unl_doors)
            all_summary = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
 
            for d in unl_doors:
                calls.append({"name": "lock_door", "args": {"door": d, "state": "lock"}})
 
            resp_parts.append(" ".join(f"{DOOR_DISPLAY[d].title()} locked." for d in unl_doors))
            part1_prompt = random.choice(door_bulk)
            part1_think = (
                f"(1) '{part1_prompt}': "
                f"Checking ALL connected doors: {all_summary}. "
                f"{len(unl_doors)} door(s) currently unlocked ({unl_names}). "
                f"Issuing {len(unl_doors)} lock_door(lock) calls."
            )
 
        # ── Part 2: Gadget Action ──
        part2_think = ""
        part2_prompt = ""
 
        if gadget_type == "speaker":
            for r in sp_rooms: state["speaker"][r] = "playing"
            state["speaker"][target_r] = "stopped"
            
            media_trace = ""
            if random.random() < 0.5:
                part2_prompt = random.choice(speaker_phrases)
                calls.append({"name": "control_speaker", "args": {"room": target_r, "action": "play"}})
                resp_parts.append(f"Playing music on the {ROOM_DISPLAY[target_r]} speaker.")
            else:
                media = random.choice(LOCAL_MUSIC)
                part2_prompt = random.choice([f"play {media}", f"put on {media}", f"start {media}"])
                calls.append({"name": "control_speaker", "args": {"room": target_r, "action": "play", "media": media}})
                resp_parts.append(f"Playing '{media}' on the {ROOM_DISPLAY[target_r]} speaker.")
                media_trace = f", media='{media}'"
            
            sp_str = ", ".join(sp_rooms)
            if len(sp_rooms) == 1:
                part2_think = (
                    f"(2) '{part2_prompt}': Checking CONNECTED SPEAKERS: [{sp_str}]. "
                    f"Exactly one speaker connected. Resolving to {target_r}. "
                    f"Calling control_speaker(room={target_r}, action=play{media_trace})."
                )
            else:
                part2_think = (
                    f"(2) '{part2_prompt}': Checking CONNECTED SPEAKERS: [{sp_str}]. "
                    f"Multiple speakers connected. Exactly ONE ({target_r}) is eligible ('stopped') for 'play'. "
                    f"Inferring {target_r}. "
                    f"Calling control_speaker(room={target_r}, action=play{media_trace})."
                )
 
        elif gadget_type == "tv":
            for r in tv_rooms: state["tv"][r] = "on"
            state["tv"][target_r] = "off"
            part2_prompt = random.choice(tv_phrases)
            calls.append({"name": "control_tv", "args": {"room": target_r, "state": "on"}})
            resp_parts.append(f"The {ROOM_DISPLAY[target_r].title()} TV is now on.")
            tv_str = ", ".join(tv_rooms)
            if len(tv_rooms) == 1:
                part2_think = (
                    f"(2) '{part2_prompt}': Checking CONNECTED TVs: [{tv_str}]. "
                    f"Exactly one TV connected. Resolving to {target_r}. "
                    f"Calling control_tv(room={target_r}, state=on)."
                )
            else:
                part2_think = (
                    f"(2) '{part2_prompt}': Checking CONNECTED TVs: [{tv_str}]. "
                    f"Multiple TVs connected. Exactly ONE ({target_r}) is eligible ('off') for 'on'. "
                    f"Inferring {target_r}. Calling control_tv(room={target_r}, state=on)."
                )
 
        else:  # fan
            for r in fan_rooms: state["fan"][r]["state"] = "on"
            state["fan"][target_r]["state"] = "off"
            part2_prompt = random.choice(fan_phrases)
            calls.append({"name": "control_fan", "args": {"room": target_r, "state": "on"}})
            resp_parts.append(f"The {ROOM_DISPLAY[target_r].title()} fan is now on.")
            fan_str = ", ".join(fan_rooms)
            if len(fan_rooms) == 1:
                part2_think = (
                    f"(2) '{part2_prompt}': Checking CONNECTED FANS: [{fan_str}]. "
                    f"Exactly one fan connected. Resolving to {target_r}. "
                    f"Calling control_fan(room={target_r}, state=on)."
                )
            else:
                part2_think = (
                    f"(2) '{part2_prompt}': Checking CONNECTED FANS: [{fan_str}]. "
                    f"Multiple fans connected. Exactly ONE ({target_r}) is eligible ('off') for 'on'. "
                    f"Inferring {target_r}. Calling control_fan(room={target_r}, state=on)."
                )
 
        prompt = random.choice([
            f"{part1_prompt.capitalize()} and {part2_prompt}.",
            f"{part2_prompt.capitalize()} and {part1_prompt}.",
        ])
 
        non_target_rooms = [r for r in avail_r if r != target_r]
        u_room = random.choice(non_target_rooms) if non_target_rooms else ""
 
        n_calls = len(calls)
        think = (
            f"Compound request — bulk {bulk_type} command and gadget action. "
            f"Evaluating each part separately. "
            f"{part1_think} "
            f"{part2_think} "
            f"Total: {n_calls} tool call{'s' if n_calls != 1 else ''} required. Emitting all {n_calls}."
        )
 
        resp = " ".join(resp_parts)
        action_log = build_distractor_log(avail_r, avail_d, n=1) if random.random() < 0.4 else ""
 
        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=u_room, action_log=action_log, think_trace=think,
            category="bulk_plus_gadget"
        ))
 
    return examples
    
def gen_double_bulk(target: int = 1_500) -> list:
    examples = []
 
    light_bulk = [
        ("on all the lights", "on"),
        ("turn on all the lights", "on"),
        ("turn off all the lights", "off"),
        ("off all the lights", "off"),
        ("kill all the lights", "off"),
        ("turn off the lights that are on", "off"),
        ("turn on the lights that are off", "on"),
    ]
    door_bulk = [
        ("close all the doors that are open", "lock"),
        ("lock all the doors", "lock"),
        ("secure every door", "lock"),
        ("close any doors that are open", "lock"),
        ("open all the doors", "unlock"),
        ("unlock all doors", "unlock"),
    ]
 
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=3)
        if len(avail_r) < 6 and random.random() < 0.5:
            pool_r = [r for r in ALL_ROOMS if r not in avail_r]
            avail_r.extend(pool_r[:6 - len(avail_r)])
 
        state = generate_random_state(avail_r, avail_d)
 
        l_prompt, l_action = random.choice(light_bulk)
        d_prompt, d_action = random.choice(door_bulk)
 
        calls = []
        resp_parts = []
 
        # ── PART 1: BULK LIGHTS ──
        l_opp = "off" if l_action == "on" else "on"
        # 20% chance: all lights already in target state (already satisfied)
        if random.random() < 0.20:
            for r in avail_r:
                state["lights"][r]["state"] = l_action   # all already correct
            act_rooms = []
        else:
            act_rooms = random.sample(avail_r, random.randint(
                max(1, len(avail_r) - 2), len(avail_r)))
            for r in avail_r:
                state["lights"][r]["state"] = l_opp if r in act_rooms else l_action
 
        act_room_names = ", ".join(ROOM_DISPLAY[r] for r in act_rooms)
        l_summary = ", ".join(f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r)
 
        for r in act_rooms:
            calls.append({"name": "toggle_lights", "args": {"room": r, "state": l_action}})
 
        if act_rooms:
            resp_parts.append(f"All {len(act_rooms)} lights turned {l_action}.")
            part1_think = (
                f"(1) '{l_prompt}': "
                f"Checking ALL connected lights: {l_summary}. "
                f"{len(act_rooms)} light(s) currently {l_opp} ({act_room_names}). "
                f"Issuing {len(act_rooms)} toggle_lights({l_action}) calls."
            )
        else:
            resp_parts.append(f"All lights are already {l_action}.")
            part1_think = (
                f"(1) '{l_prompt}': "
                f"Checking ALL connected lights: {l_summary}. "
                f"All lights already {l_action}."
            )
 
        # ── PART 2: BULK DOORS ──
        d_opp = "unlocked" if d_action == "lock" else "locked"
        if random.random() < 0.20:
            for d in avail_d:
                state["doors"][d] = "locked" if d_action == "lock" else "unlocked"
            act_doors = []
        else:
            act_doors = random.sample(avail_d, random.randint(
                max(1, len(avail_d) - 2), len(avail_d)))
            for d in avail_d:
                state["doors"][d] = d_opp if d in act_doors else (
                    "locked" if d_action == "lock" else "unlocked")
 
        act_door_names = ", ".join(DOOR_DISPLAY[d] for d in act_doors)
        d_aw = "locked" if d_action == "lock" else "unlocked"
        d_summary = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
 
        for d in act_doors:
            calls.append({"name": "lock_door", "args": {"door": d, "state": d_action}})
 
        if act_doors:
            resp_parts.append(f"All {len(act_doors)} doors {d_aw}.")
            part2_think = (
                f"(2) '{d_prompt}': "
                f"Checking ALL connected doors: {d_summary}. "
                f"{len(act_doors)} door(s) currently {d_opp} ({act_door_names}). "
                f"Issuing {len(act_doors)} lock_door({d_action}) calls."
            )
        else:
            resp_parts.append(f"All doors are already {d_aw}.")
            part2_think = (
                f"(2) '{d_prompt}': "
                f"Checking ALL connected doors: {d_summary}. "
                f"All doors already {d_aw}."
            )
 
        n_calls = len(calls)
        if n_calls == 0 and len(resp_parts) == 2:
            # Both scopes already satisfied — valid training example
            pass   # allow it through
        elif n_calls < 2 and not (act_rooms == [] and act_doors == []):
            continue   # skip genuinely degenerate single-call cases
 
        prompt = apply_typo(random.choice([
            f"{l_prompt.capitalize()} and {d_prompt}.",
            f"{d_prompt.capitalize()} and {l_prompt}.",
            f"Can you {l_prompt} and {d_prompt}?",
        ]))
 
        think = (
            f"Compound request — two INDEPENDENT 'and'-linked bulk commands. "
            f"Evaluating each scope separately. "
            f"{part1_think} "
            f"{part2_think} "
            f"Total: {n_calls} tool call{'s' if n_calls != 1 else ''} required. Emitting all {n_calls}."
        )
 
        resp = " ".join(resp_parts)
        u_room = random.choice(["", random.choice(avail_r)])
 
        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=u_room,
            think_trace=think, category="double_bulk_stress"
        ))
 
    return examples
    
def gen_tv_commands(target: int = 1_000) -> list:
    examples = []
    while len(examples) < target:
        avail_r, avail_d = sample_topology()
        state    = generate_random_state(avail_r, avail_d)
        tv_rooms = [r for r in avail_r if r in TV_ROOMS]
        if not tv_rooms: continue
        desired    = random.choice(["on", "off"])
        tmpls      = TV_ON_TMPLS if desired == "on" else TV_OFF_TMPLS
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.5 else ""
        def already_sat(st, r): return st["tv"].get(r) == desired
        def make_call(r, d):    return {"name": "control_tv", "args": {"room": r, "state": d}}
        def resp_fn(alias, d):  return f"The {alias} TV is now {d}."
        def clarify(rooms):     return (f"Which TV? I have TVs in: "
                                         f"{', '.join(ROOM_DISPLAY[r] for r in rooms)}.")
        ex = _resolve_gadget(avail_r, avail_d, state, "tv", tv_rooms, desired, tmpls,
            TV_ON_EXPLICIT, TV_OFF_EXPLICIT, already_sat, make_call, resp_fn, clarify,
            "tv_commands", action_log=action_log)
        if ex: examples.append(ex)
    return examples


def gen_speaker_commands(target: int = 2_500) -> list:
    """GAP G/H: speaker action resolution with full synonym coverage and media."""
    examples = []
    ACTION_MAP = {
        "play":     (SPEAKER_PLAY_TMPLS,  SPEAKER_PLAY_EXPLICIT,  SPEAKER_PLAY_EXPLICIT),
        "pause":    (SPEAKER_PAUSE_TMPLS, SPEAKER_PAUSE_EXPLICIT, SPEAKER_PAUSE_EXPLICIT),
        "stop":     (SPEAKER_STOP_TMPLS,  SPEAKER_STOP_EXPLICIT,  SPEAKER_STOP_EXPLICIT),
        "next":     (SPEAKER_NEXT_TMPLS,  SPEAKER_NEXT_EXPLICIT,  SPEAKER_NEXT_EXPLICIT),
        "previous": (SPEAKER_PREV_TMPLS,  SPEAKER_PREV_EXPLICIT,  SPEAKER_PREV_EXPLICIT),
    }
    ALREADY_SAT    = {"play": "playing", "pause": "paused", "stop": "stopped"}
    ELIGIBLE_STATE = {"play": "stopped", "pause": "playing", "stop": "playing",
                      "next": None, "previous": None}
    ACTION_WEIGHTS = [20, 20, 40, 12, 8]

    while len(examples) < target:
        avail_r, avail_d = sample_topology()
        state    = generate_random_state(avail_r, avail_d)
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not sp_rooms: continue
        action   = random.choices(["play","pause","stop","next","previous"],
                                  weights=ACTION_WEIGHTS)[0]
                                  
        media_val = None
        
        # Dynamically generate media and templates for 'play'
        if action == "play" and random.random() < 0.5:
            media_val = random.choice(LOCAL_MUSIC)
            tmpls = [f"Play {media_val}.", f"Can you play {media_val}?", f"Put on {media_val}."]
            ex_on = [f"Play {media_val} in the {{r}}.", f"Put on {media_val} in the {{r}}."]
            ex_off = ex_on
        else:
            tmpls, ex_on, ex_off = ACTION_MAP[action]
            
        sat_state = ALREADY_SAT.get(action)
        eligible  = ELIGIBLE_STATE[action]
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.6 else ""

        def already_sat(st, r, _sat=sat_state):
            return _sat is not None and st["speaker"].get(r) == _sat

        def make_call(r, a): return {"name": "control_speaker",
                                      "args": {"room": r, "action": a}}

        def resp_fn(alias, a):
            if a == "next":     return f"Skipping to the next track on the {alias} speaker."
            if a == "previous": return f"Playing the previous track on the {alias} speaker."
            if a == "play":     
                return f"Playing '{media_val}' on the {alias} speaker." if media_val else f"Playing music on the {alias} speaker."
            if a == "pause":    return f"Paused the {alias} speaker."
            return f"Stopped the music on the {alias} speaker."

        def clarify(rooms): return (f"Which speaker? I have them in: "
                                     f"{', '.join(ROOM_DISPLAY[r] for r in rooms)}.")

        weights = [20, 20, 20, 20, 20] if action in ("play","pause","stop") \
            else [30, 30, 30, 5, 5]
            
        ex = _resolve_gadget(
            avail_r, avail_d, state, "speaker", sp_rooms, action, tmpls,
            ex_on, ex_off, already_sat, make_call, resp_fn, clarify,
            "speaker_commands", scenario_weights=weights,
            action_log=action_log, eligible_state_for_infer=eligible, media=media_val)
            
        if ex: examples.append(ex)
    return examples


def gen_fan_commands(target: int = 1_000) -> list:
    examples = []
    while len(examples) < target:
        avail_r, avail_d = sample_topology()
        state     = generate_random_state(avail_r, avail_d)
        fan_rooms = [r for r in avail_r if r in FAN_ROOMS]
        if not fan_rooms: continue
        if random.random() < 0.20:
            r = random.choice(fan_rooms); alias = random.choice(ROOM_ALIASES[r])
            speed = random.choice(["low","medium","high"])
            tmpl  = random.choice(FAN_SPEED_TMPLS).format(r=alias, sp=speed)
            think = (f"User requested fan in {r} at speed {speed}. "
                     f"Calling control_fan.")
            examples.append(build_ex(tmpl,
                [{"name": "control_fan", "args": {"room": r, "state": "on", "speed": speed}}],
                f"The {alias} fan is now on at {speed} speed.", avail_r, avail_d, state,
                think_trace=think, category="fan_commands"))
            continue
        desired    = random.choice(["on","off"])
        tmpls      = FAN_ON_TMPLS if desired == "on" else FAN_OFF_TMPLS
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.5 else ""
        def already_sat(st, r): return st["fan"].get(r, {}).get("state") == desired
        def make_call(r, d):    return {"name": "control_fan", "args": {"room": r, "state": d}}
        def resp_fn(alias, d):  return f"The {alias} fan is now {d}."
        def clarify(rooms):     return (f"Which fan? I have fans in: "
                                         f"{', '.join(ROOM_DISPLAY[r] for r in rooms)}.")
        ex = _resolve_gadget(avail_r, avail_d, state, "fan", fan_rooms, desired, tmpls,
            FAN_ON_EXPLICIT, FAN_OFF_EXPLICIT, already_sat, make_call, resp_fn, clarify,
            "fan_commands", action_log=action_log)
        if ex: examples.append(ex)
    return examples


def gen_compound_local(target: int = 1_500) -> list:
    """
    Closes the 'Local Compound Blind Spot': compound commands where one or
    both actions use implicit local resolution via current_user_room.

    Covers:
    A) both_implicit:               "off the light and close the door"
    B) explicit_light+implicit_door: "on the kitchen light and close this door"
    C) explicit_door+implicit_light: "open the office door and on the light"
    D) implicit_light+gadget:        "on this light and play some music"
    E) scene+implicit_light:         "movie night and on the light"
    F) scene+implicit_door:          "bedtime and close this door"
    """
    examples = []
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]

    on_light_p  = ["on the light", "turn on the light", "on this light",
                   "switch on the light", "light on"]
    off_light_p = ["off the light", "turn off the light", "off this light",
                   "kill the light", "light off"]
    open_door_p = ["open the door", "open this door", "unlock this door",
                   "unlock the door"]
    close_door_p = ["close the door", "close this door", "lock this door",
                    "lock the door", "shut the door", "shut this door"]

    case_weights = [35, 20, 15, 15, 10, 5]
    cases = ["A", "B", "C", "D", "E", "F"]

    attempts = 0
    while len(examples) < target and attempts < target * 5:
        attempts += 1
        case = random.choices(cases, weights=case_weights)[0]

        if case == "A":
            u_room = random.choice(ROOM_DOOR_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[u_room],
                                               required_doors=[u_room])
            state = generate_random_state(avail_r, avail_d)
            ls = random.choice(["on", "off"])
            ds = random.choice(["lock", "unlock"])
            aw = "locked" if ds == "lock" else "unlocked"
            apply_force(state, {"lights": {u_room: "off" if ls == "on" else "on"},
                                "doors":  {u_room: "unlocked" if ds == "lock" else "locked"}},
                        avail_r, avail_d)
            l_p = random.choice(on_light_p if ls == "on" else off_light_p)
            d_p = random.choice(open_door_p if ds == "unlock" else close_door_p)
            r_alias = random.choice(ROOM_ALIASES[u_room])
            d_alias = random.choice(DOOR_ALIASES[u_room])
            prompt = random.choice([
                f"{l_p.capitalize()} and {d_p}.",
                f"{d_p.capitalize()} and {l_p}.",
                f"Can you {l_p} and {d_p}?",
                f"Please {l_p} and {d_p}.",
            ])
            calls = [{"name": "toggle_lights", "args": {"room": u_room, "state": ls}},
                     {"name": "lock_door",     "args": {"door": u_room, "state": ds}}]
            resp = f"{r_alias.title()} light {ls} and {d_alias} {aw}."
            think = (
                f"Compound request with two implicit local references. "
                f"Sub-action 1: '{l_p}' — current_user_room='{u_room}' → "
                f"toggle_lights(room={u_room}, state={ls}). "
                f"Sub-action 2: '{d_p}' — current_user_room='{u_room}' → "
                f"lock_door(door={u_room}, state={ds}). "
                f"Both must be called independently."
            )
            action_log = build_distractor_log(avail_r, avail_d, n=1) \
                if random.random() < 0.3 else ""
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=u_room, action_log=action_log,
                think_trace=think, category="compound_local"))

        elif case == "B":
            u_room = random.choice(ROOM_DOOR_ROOMS)
            other_rooms = [r for r in ALL_ROOMS if r != u_room]
            if not other_rooms: continue
            l_room = random.choice(other_rooms)
            avail_r, avail_d = sample_topology(required_rooms=[l_room, u_room],
                                               required_doors=[u_room])
            state = generate_random_state(avail_r, avail_d)
            ls = random.choice(["on", "off"])
            ds = random.choice(["lock", "unlock"])
            aw = "locked" if ds == "lock" else "unlocked"
            apply_force(state, {"lights": {l_room: "off" if ls == "on" else "on"},
                                "doors":  {u_room: "unlocked" if ds == "lock" else "locked"}},
                        avail_r, avail_d)
            l_alias = random.choice(ROOM_ALIASES[l_room])
            d_alias = random.choice(DOOR_ALIASES[u_room])
            d_p = random.choice(open_door_p if ds == "unlock" else close_door_p)
            l_verb = "on" if ls == "on" else "off"
            prompt = random.choice([
                f"{l_verb} the {l_alias} light and {d_p}.",
                f"{d_p.capitalize()} and turn {l_verb} the {l_alias} light.",
                f"Turn {l_verb} the {l_alias} light and {d_p}.",
            ])
            calls = [{"name": "toggle_lights", "args": {"room": l_room, "state": ls}},
                     {"name": "lock_door",     "args": {"door": u_room, "state": ds}}]
            resp = f"{l_alias.title()} light {ls} and {d_alias} {aw}."
            think = (
                f"Compound request. "
                f"Sub-action 1: '{l_alias} light' is explicitly named → "
                f"toggle_lights(room={l_room}, state={ls}). "
                f"Sub-action 2: '{d_p}' — current_user_room='{u_room}' → lock_door(door={u_room}, state={ds}). "
                f"Both must be called."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=u_room, think_trace=think, category="compound_local"))

        elif case == "C":
            u_room = random.choice(ROOM_DOOR_ROOMS)
            other_doors = [d for d in ALL_DOORS if d != u_room]
            if not other_doors: continue
            exp_door = random.choice(other_doors)
            avail_r, avail_d = sample_topology(required_rooms=[u_room],
                                               required_doors=[exp_door, u_room])
            state = generate_random_state(avail_r, avail_d)
            ls = random.choice(["on", "off"])
            ds = random.choice(["lock", "unlock"])
            aw = "locked" if ds == "lock" else "unlocked"
            apply_force(state, {"lights": {u_room: "off" if ls == "on" else "on"},
                                "doors":  {exp_door: "unlocked" if ds == "lock" else "locked"}},
                        avail_r, avail_d)
            r_alias  = random.choice(ROOM_ALIASES[u_room])
            d_alias  = random.choice(DOOR_ALIASES[exp_door])
            l_p      = random.choice(on_light_p if ls == "on" else off_light_p)
            d_verb   = random.choice(["open", "unlock"]) if ds == "unlock" \
                       else random.choice(["close", "lock"])
            prompt = random.choice([
                f"{l_p.capitalize()} and {d_verb} the {d_alias}.",
                f"{d_verb.capitalize()} the {d_alias} and {l_p}.",
            ])
            calls = [{"name": "toggle_lights", "args": {"room": u_room,   "state": ls}},
                     {"name": "lock_door",     "args": {"door": exp_door, "state": ds}}]
            resp = f"{r_alias.title()} light {ls} and {d_alias} {aw}."
            think = (
                f"Compound request. "
                f"Sub-action 1: '{l_p}' — current_user_room='{u_room}' → "
                f"toggle_lights(room={u_room}, state={ls}). "
                f"Sub-action 2: '{d_alias}' is explicitly named → lock_door(door={exp_door}, state={ds}). "
                f"Both must be called."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=u_room, think_trace=think, category="compound_local"))

        elif case == "D":
            u_room = random.choice(ALL_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[u_room])
            state = generate_random_state(avail_r, avail_d)
            ls = random.choice(["on", "off"])
            apply_force(state, {"lights": {u_room: "off" if ls == "on" else "on"}},
                        avail_r, avail_d)
            r_alias = random.choice(ROOM_ALIASES[u_room])
            l_p = random.choice(on_light_p if ls == "on" else off_light_p)

            gadget = random.choice(["speaker", "tv"])
            g_p = g_think = None
            g_calls = []
            g_resp = ""

            if gadget == "speaker":
                sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
                if not sp_rooms: continue
                if len(sp_rooms) == 1:
                    sp_r = sp_rooms[0]
                    state["speaker"][sp_r] = "stopped"
                    sp_alias = random.choice(ROOM_ALIASES[sp_r])
                    g_p = random.choice(["play some music", "play music",
                                         "start the music", "on the speaker"])
                    g_calls = [{"name": "control_speaker",
                                "args": {"room": sp_r, "action": "play"}}]
                    g_resp = f"Playing music on the {sp_alias} speaker."
                    g_think = (f"'{g_p}' → only one speaker connected ({sp_r}), "
                               f"Rule 1 → control_speaker(room={sp_r}, action=play).")
                elif u_room in sp_rooms:
                    sp_r = u_room
                    state["speaker"][sp_r] = "stopped"
                    sp_alias = random.choice(ROOM_ALIASES[sp_r])
                    g_p = random.choice(["play some music", "play music", "on the speaker"])
                    g_calls = [{"name": "control_speaker",
                                "args": {"room": sp_r, "action": "play"}}]
                    g_resp = f"Playing music on the {sp_alias} speaker."
                    g_think = (f"'{g_p}' → current_user_room='{u_room}' "
                               f"has speaker, Rule 2 → control_speaker(room={u_room}, action=play).")
                else:
                    continue
            else:  # TV
                tv_rooms = [r for r in avail_r if r in TV_ROOMS]
                if not tv_rooms: continue
                if len(tv_rooms) == 1:
                    tv_r = tv_rooms[0]
                    state["tv"][tv_r] = "off"
                    tv_alias = random.choice(ROOM_ALIASES[tv_r])
                    g_p = "on the TV"
                    g_calls = [{"name": "control_tv",
                                "args": {"room": tv_r, "state": "on"}}]
                    g_resp = f"{tv_alias.title()} TV on."
                    g_think = (f"'on the TV' → only one TV connected ({tv_r}), "
                               f"Rule 1 → control_tv(room={tv_r}, state=on).")
                elif u_room in tv_rooms:
                    tv_r = u_room
                    state["tv"][tv_r] = "off"
                    tv_alias = random.choice(ROOM_ALIASES[tv_r])
                    g_p = "on the TV"
                    g_calls = [{"name": "control_tv",
                                "args": {"room": tv_r, "state": "on"}}]
                    g_resp = f"{tv_alias.title()} TV on."
                    g_think = (f"'on the TV' → current_user_room='{u_room}' has TV, "
                               f"Rule 2 → control_tv(room={u_room}, state=on).")
                else:
                    continue

            prompt = random.choice([
                f"{l_p.capitalize()} and {g_p}.",
                f"{g_p.capitalize()} and {l_p}.",
                f"Can you {l_p} and {g_p}?",
            ])
            calls = [{"name": "toggle_lights",
                      "args": {"room": u_room, "state": ls}}] + g_calls
            resp = f"{r_alias.title()} light {ls}. {g_resp}"
            think = (
                f"Compound request. Sub-action 1: '{l_p}' → "
                f"parsed direction: '{ls}'. current_user_room='{u_room}' → "
                f"toggle_lights(room={u_room}, state={ls}). "
                f"Sub-action 2: {g_think} "
                f"Both must be called."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=u_room, think_trace=think, category="compound_local"))

        elif case == "E":
            u_room = random.choice(ALL_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[u_room])
            state = generate_random_state(avail_r, avail_d)
            scene = random.choice(SCENES)
            strig = random.choice(SCENE_TRIGGERS[scene])
            ls = random.choice(["on", "off"])
            apply_force(state, {"lights": {u_room: "off" if ls == "on" else "on"}},
                        avail_r, avail_d)
            state["active_scene"] = None
            r_alias = random.choice(ROOM_ALIASES[u_room])
            l_p = random.choice(on_light_p if ls == "on" else off_light_p)
            prompt = random.choice([
                f"{strig.capitalize()} and {l_p}.",
                f"{l_p.capitalize()} and {strig}.",
                f"{strig.capitalize()} please and {l_p}.",
            ])
            calls = [{"name": "set_scene",     "args": {"scene": scene}},
                     {"name": "toggle_lights", "args": {"room": u_room, "state": ls}}]
            resp = f"{SCENE_RESP[scene]} {r_alias.title()} light {ls}."
            think = (
                f"Compound request. "
                f"Sub-action 1: '{strig}' → set_scene(scene={scene}). "
                f"Sub-action 2: '{l_p}' → current_user_room='{u_room}' → "
                f"toggle_lights(room={u_room}, state={ls}). "
                f"Both must be called."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=u_room, think_trace=think, category="compound_local"))

        elif case == "F":
            u_room = random.choice(ROOM_DOOR_ROOMS)
            avail_r, avail_d = sample_topology(required_rooms=[u_room],
                                               required_doors=[u_room])
            state = generate_random_state(avail_r, avail_d)
            scene = random.choice(SCENES)
            strig = random.choice(SCENE_TRIGGERS[scene])
            ds = random.choice(["lock", "unlock"])
            aw = "locked" if ds == "lock" else "unlocked"
            apply_force(state, {"doors": {u_room: "unlocked" if ds == "lock" else "locked"}},
                        avail_r, avail_d)
            state["active_scene"] = None
            d_alias = random.choice(DOOR_ALIASES[u_room])
            d_p = random.choice(open_door_p if ds == "unlock" else close_door_p)
            prompt = random.choice([
                f"{strig.capitalize()} and {d_p}.",
                f"{d_p.capitalize()} and {strig}.",
            ])
            calls = [{"name": "set_scene",  "args": {"scene": scene}},
                     {"name": "lock_door", "args": {"door": u_room, "state": ds}}]
            resp = f"{SCENE_RESP[scene]} {d_alias.title()} {aw}."
            think = (
                f"Compound request. "
                f"Sub-action 1: '{strig}' → set_scene(scene={scene}). "
                f"Sub-action 2: '{d_p}' → current_user_room='{u_room}' → "
                f"lock_door(door={u_room}, state={ds}). "
                f"Both must be called."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=u_room, think_trace=think, category="compound_local"))

    return examples[:target]



def gen_action_log_gadgets(target: int = 1_000) -> list:
    """
    Trains pronoun 'it'/'them' resolution for TV/Speaker/Fan from action log.
    Closes the gap where model hallucinated wrong device after a gadget action.
    (Fixes turn 160: 'off it' after speaker played → model guessed door.)
    """
    examples = []
    pronoun_off = ["Off it.", "Turn it off.", "Stop it.", "Kill it.",
                   "Shut it off.", "Switch it off.", "Cut it."]
    pronoun_on  = ["On it.", "Turn it on.", "Switch it on.", "Put it back on.",
                   "Bring it back on.", "On it please."]

    while len(examples) < target:
        avail_r, avail_d = sample_topology()
        state  = generate_random_state(avail_r, avail_d)
        device = random.choice(["tv", "speaker", "fan"])

        if device == "tv":
            valid = [r for r in avail_r if r in TV_ROOMS]
        elif device == "speaker":
            valid = [r for r in avail_r if r in SPEAKER_ROOMS]
        else:
            valid = [r for r in avail_r if r in FAN_ROOMS]
        if not valid: continue

        r = random.choice(valid)
        primary_mins = random.randint(1, 3)
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"

        if device == "tv":
            prior  = random.choice(["on", "off"])
            target_s = "off" if prior == "on" else "on"
            state["tv"][r] = prior
            log    = fmt_txn(primary_mins, [f"control_tv(room={r}, state={prior})"],
                             f"{ROOM_DISPLAY[r]} TV turned {prior}.")
            phrase = random.choice(pronoun_off if target_s == "off" else pronoun_on)
            calls  = [{"name": "control_tv", "args": {"room": r, "state": target_s}}]
            resp   = f"The {ROOM_DISPLAY[r]} TV is now {target_s}."
            think  = (
                f"User said '{phrase}'. "
                f"Pronoun 'it' → first [...] block ({t_label}): {ROOM_DISPLAY[r]} TV. "
                f"Current state: '{prior}'. User wants: '{target_s}' (opposite). "
                f"Calling control_tv(room={r}, state={target_s})."
            )

        elif device == "speaker":
            opts = [("play",  "stop",  "playing", pronoun_off),
                    ("play",  "pause", "playing", pronoun_off),
                    ("stop",  "play",  "stopped", pronoun_on),
                    ("pause", "play",  "paused",  pronoun_on)]
            prior_a, target_a, prior_sp, phrase_list = random.choice(opts)
            state["speaker"][r] = prior_sp
            log   = fmt_txn(primary_mins,
                            [f"control_speaker(room={r}, action={prior_a})"],
                            f"{ROOM_DISPLAY[r]} speaker: {prior_a}.")
            phrase = random.choice(phrase_list)
            calls  = [{"name": "control_speaker",
                       "args": {"room": r, "action": target_a}}]
            if   target_a == "play":  resp = f"Playing music on the {ROOM_DISPLAY[r]} speaker."
            elif target_a == "pause": resp = f"Paused the {ROOM_DISPLAY[r]} speaker."
            else:                     resp = f"Stopped the music on the {ROOM_DISPLAY[r]} speaker."
            think = (
                f"User said '{phrase}'. "
                f"Pronoun 'it' → first [...] block ({t_label}): {ROOM_DISPLAY[r]} speaker. "
                f"Last logged action was '{prior_a}'. Calling opposite: "
                f"control_speaker(room={r}, action={target_a})."
            )

        else:  # fan
            prior  = random.choice(["on", "off"])
            target_s = "off" if prior == "on" else "on"
            state["fan"][r]["state"] = prior
            log    = fmt_txn(primary_mins, [f"control_fan(room={r}, state={prior})"],
                             f"{ROOM_DISPLAY[r]} fan turned {prior}.")
            phrase = random.choice(pronoun_off if target_s == "off" else pronoun_on)
            calls  = [{"name": "control_fan", "args": {"room": r, "state": target_s}}]
            resp   = f"The {ROOM_DISPLAY[r]} fan is now {target_s}."
            think  = (
                f"User said '{phrase}'. "
                f"Pronoun 'it' → first [...] block ({t_label}): {ROOM_DISPLAY[r]} fan. "
                f"Current state: '{prior}'. User wants: '{target_s}' (opposite). "
                f"Calling control_fan(room={r}, state={target_s})."
            )

        dist = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2),
                                    start_mins=primary_mins + random.randint(7, 14))
        action_log = log + "\n" + dist if random.random() < 0.6 else log

        # Vary user_room to reinforce that pronoun ignores current_user_room
        other = [x for x in avail_r if x != r]
        user_room = random.choice(other) if other and random.random() < 0.5 else ""

        examples.append(build_ex(phrase, calls, resp, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="action_log_gadgets"))

    return examples

# ══════════════════════════════════════════════════════════════════════
# v12 NEW / FIXED GENERATORS
# ══════════════════════════════════════════════════════════════════════

def gen_back_synonym_disambiguation(target: int = 1_200) -> list:
    examples = []
    single_phrases = [
        "On it back.", "Turn it back on.", "Put it back on.",
        "Bring it back.", "Switch it back on.","On it",
    ]
    multi_phrases = [
        "On them back.", "Turn them back on.", "Put them back on.",
        "On those lights back.", "Turn those lights back on.",
        "Bring the lights back.", "Bring them back on.",
        "On those back.", "Lights back on.", "Turn those back on.",
    ]
    for _ in range(target):
        is_multi = random.random() < 0.55
        avail_r, avail_d = sample_topology(min_rooms=2)
        state = generate_random_state(avail_r, avail_d)
        primary_mins = random.randint(1, 4)
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"

        if is_multi:
            n_rooms = random.choices([2, 3, 4, 5], weights=[20, 30, 30, 20])[0]
            rooms = random.sample(avail_r, min(n_rooms, len(avail_r)))
            if len(rooms) < 2: continue
            
            # ABSOLUTE GROUNDING: state is forced to "off" so toggling to "on" is required
            for r in rooms:
                apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)

            call_strs = [f"toggle_lights(room={r}, state=off)" for r in rooms]
            rooms_list = ", ".join(ROOM_DISPLAY[r] for r in rooms)
            
            # FIX: Period-separated formatting to match production logs exactly
            summary = " ".join(f"{ROOM_DISPLAY[r].title()} light turned off." for r in rooms)
            
            primary_txn = fmt_txn(primary_mins, call_strs, summary)

            prompt = random.choice(multi_phrases)
            calls  = [{"name": "toggle_lights", "args": {"room": r, "state": "on"}} for r in rooms]
            resp   = " ".join(f"{ROOM_DISPLAY[r].title()} light on." for r in rooms)
            think  = (
                f"User said '{prompt}'. "
                f"'back'/'on them back'/'those lights back' signals reverting the previous "
                f"light states, evidenced by the first [...] block ({t_label}) which records "
                f"{len(rooms)} lights ({rooms_list}) being turned off. "
                f"The context establishes this strictly as a lighting reversion command. "
                f"Restoring all {len(rooms)} lights to 'on'. "
                f"Issuing {len(rooms)} toggle_lights calls."
            )
        else:
            r = random.choice(avail_r)
            apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
            primary_txn = fmt_txn(primary_mins,
                                  [f"toggle_lights(room={r}, state=off)"],
                                  f"{ROOM_DISPLAY[r]} light turned off.")
            alias  = random.choice(ROOM_ALIASES[r])
            prompt = random.choice(single_phrases)
            calls  = [{"name": "toggle_lights", "args": {"room": r, "state": "on"}}]
            resp   = f"The {alias} light is now on."
            think  = (
                f"User said '{prompt}'. "
                f"'back'/'on it back' signals reverting the previous light state, "
                f"evidenced by the first [...] block ({t_label}). "
                f"The context establishes this strictly as a lighting reversion command. "
                f"Restoring the {ROOM_DISPLAY[r]} light to 'on'. "
                f"Calling toggle_lights(room={r}, state=on)."
            )

        dist = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2),
                                    start_mins=primary_mins + random.randint(8, 14))
        action_log = primary_txn + "\n" + dist
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think, category="disambiguate_back"))
    return examples

def gen_rule3_inference(target: int = 2_000) -> list:
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=4)
        state = generate_random_state(avail_r, avail_d)
        device = random.choice(["tv", "speaker", "fan"])

        if device == "tv":
            conn = [r for r in avail_r if r in TV_ROOMS]
            if len(conn) < 2: continue
            r_target = conn[0]
            action = random.choice(["on", "off"])
            el_state = "off" if action == "on" else "on"
            non_el   = "on"  if action == "on" else "off"
            for r in conn: state["tv"][r] = non_el
            state["tv"][r_target] = el_state
            prompt = random.choice(TV_ON_TMPLS if action == "on" else TV_OFF_TMPLS)
            call = {"name": "control_tv", "args": {"room": r_target, "state": action}}
            resp = f"The {ROOM_DISPLAY[r_target]} TV is now {action}."

        elif device == "speaker":
            conn = [r for r in avail_r if r in SPEAKER_ROOMS]
            if len(conn) < 2: continue
            r_target = conn[0]
            # FIX-C: added "previous"
            action = random.choice(["play", "pause", "stop", "next", "previous"])
            el_state = "stopped" if action in ("play",) else "playing"
            non_el   = "playing" if action in ("play",) else "stopped"
            for r in conn: state["speaker"][r] = non_el
            state["speaker"][r_target] = el_state
            if action == "play":
                prompt = random.choice(SPEAKER_PLAY_TMPLS)
            elif action == "pause":
                prompt = random.choice(SPEAKER_PAUSE_TMPLS)
            elif action == "stop":
                prompt = random.choice(SPEAKER_STOP_TMPLS)
            elif action == "next":
                prompt = random.choice(SPEAKER_NEXT_TMPLS)
            else:  # previous
                prompt = random.choice(SPEAKER_PREV_TMPLS)
            call = {"name": "control_speaker",
                    "args": {"room": r_target, "action": action}}
            if action == "play":
                resp = f"Playing music on the {ROOM_DISPLAY[r_target]} speaker."
            elif action == "pause":
                resp = f"Paused the {ROOM_DISPLAY[r_target]} speaker."
            elif action == "stop":
                resp = f"Stopped the music on the {ROOM_DISPLAY[r_target]} speaker."
            elif action == "next":
                resp = f"Skipping to the next track on the {ROOM_DISPLAY[r_target]} speaker."
            else:
                resp = f"Playing the previous track on the {ROOM_DISPLAY[r_target]} speaker."

        else:  # fan
            conn = [r for r in avail_r if r in FAN_ROOMS]
            if len(conn) < 2: continue
            r_target = conn[0]
            action = random.choice(["on", "off"])
            el_state = "off" if action == "on" else "on"
            non_el   = "on"  if action == "on" else "off"
            for r in conn: state["fan"][r]["state"] = non_el
            state["fan"][r_target]["state"] = el_state
            prompt = random.choice(FAN_ON_TMPLS if action == "on" else FAN_OFF_TMPLS)
            call = {"name": "control_fan",
                    "args": {"room": r_target, "state": action}}
            resp = f"The {ROOM_DISPLAY[r_target]} fan is now {action}."

        user_r = random.choice([r for r in avail_r if r not in conn] + [""])
        action_log = (build_distractor_log(avail_r, avail_d, n=1)
                      if random.random() < 0.5 else "")

        room_clause = (f"User is in '{user_r}' which has no {device}."
                       if user_r else "User's location is unknown.")
        conn_str = ", ".join(conn)
        think = (
            f"User requested '{action}' for the {device}. "
            f"Multiple {device}s connected: {conn_str}."
            f"Checking states: exactly ONE {device} ({r_target}) is in the "
            f"eligible state for '{action}'. "
            f"Inferring {r_target}."
        )
        examples.append(build_ex(prompt, [call], resp, avail_r, avail_d, state,
            user_room=user_r, action_log=action_log,
            think_trace=think, category="rule3_inference"))

    return examples

def gen_relative_state_clauses(target: int = 2_000) -> list:
    examples = []

    light_on_phrases  = [
        "off the light that is on", "turn off the one that is on",
        "kill the light that's on", "switch off the light that is currently on",
        "off the light that's currently on",
    ]
    light_off_phrases = [
        "on the light that is off", "turn on the one that is off",
        "switch on the light that is currently off", "on the light that's off",
    ]
    door_open_phrases  = [   
        "close the door that is open", "lock the door that is unlocked",
        "close the door that's open", "shut the door that is open",
        "lock any door that is open", "close any door that is open",
        "close the one that is open",
    ]
    door_locked_phrases = [  
        "open the door that is locked", "unlock the door that is locked",
        "open the door that's locked", "unlock any door that is locked",
        "open the one that is locked",
    ]

    light_target = int(target * 0.55)
    for idx in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)

        if idx < light_target:
            target_s = random.choice(["on", "off"])
            action_s = "off" if target_s == "on" else "on"
            n_targets    = random.randint(1, len(avail_r))
            target_rooms = random.sample(avail_r, n_targets)
            for r in avail_r: state["lights"][r]["state"] = action_s
            for r in target_rooms: state["lights"][r]["state"] = target_s
            others    = [r for r in avail_r if r not in target_rooms]
            user_room = random.choice(others) if others else random.choice(avail_r)
            prompt    = random.choice(light_on_phrases if target_s == "on" else light_off_phrases)
            names     = ", ".join(ROOM_DISPLAY[r] for r in target_rooms)
            calls     = [{"name": "toggle_lights", "args": {"room": r, "state": action_s}} for r in target_rooms]
            resp      = " ".join(f"{ROOM_DISPLAY[r].title()} light {action_s}." for r in target_rooms)
            
            # COMPLETELY SCRUBBED: No mention of user_room. Jumps straight into the logic.
            think = (
                f"User said '{prompt}'. "
                f"Relative state clause resolves directly based on STATE. "
                f"Checking STATE: {n_targets} light(s) ({names}) match light={target_s}. "
                f"Issuing {n_targets} toggle_lights(state={action_s}) call(s)."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="relative_clause"))
        else:
            target_d = random.choice(["unlocked", "locked"])
            action_d = "lock" if target_d == "unlocked" else "unlock"
            result_d = "locked" if action_d == "lock" else "unlocked"
            opp_d    = "locked" if target_d == "unlocked" else "unlocked"
            n_targets    = random.randint(1, len(avail_d))
            target_doors = random.sample(avail_d, n_targets)
            for d in avail_d: state["doors"][d] = opp_d
            for d in target_doors: state["doors"][d] = target_d
            user_room = random.choice(avail_r)
            prompt    = random.choice(door_open_phrases if target_d == "unlocked" else door_locked_phrases)
            names     = ", ".join(DOOR_DISPLAY[d] for d in target_doors)
            calls     = [{"name": "lock_door", "args": {"door": d, "state": action_d}} for d in target_doors]
            resp      = " ".join(f"{DOOR_DISPLAY[d].title()} {result_d}." for d in target_doors)
            
            # COMPLETELY SCRUBBED: No mention of user_room.
            think = (
                f"User said '{prompt}'. "
                f"Relative state clause resolves directly based on STATE. "
                f"Checking STATE: {n_targets} door(s) ({names}) match state={target_d}. "
                f"Issuing {n_targets} lock_door(state={action_d}) call(s)."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                user_room=user_room, think_trace=think, category="relative_clause"))
    return examples



 
# ══════════════════════════════════════════════════════════════════════
# FIX-V13-A  gen_state_grounding_stress
# ══════════════════════════════════════════════════════════════════════
 
def gen_state_grounding_stress(target: int = 3_000) -> list:
    """
    FIX-V13-A: Closes state hallucination (Bucket 2).
 
    Key design:
    • Think trace always opens with "Checking STATE: <room>=<literal_value>."
      and closes with "Match — …" or "Mismatch — …" before any decision.
    • 50% match / 50% mismatch split at generation time (not random.choice)
      so the model sees both sides of the boundary equally.
    • Includes door variants so the same read-before-act habit fires for
      all device types.
    • No action log — keeps the signal clean.
    """
    examples = []
 
    on_phrases  = [
        "Turn on the {r} light.", "On the {r} light.", "Switch on the {r} light.",
        "Can you turn on the {r} light?", "{r} light on.", "I need the {r} light on.",
        "Put on the {r} light.", "Let there be light in the {r}.",
    ]
    off_phrases = [
        "Turn off the {r} light.", "Off the {r} light.", "Kill the {r} light.",
        "Switch off the {r} light.", "{r} light off.", "Cut the {r} light.",
        "Lights off in the {r}.", "I need the {r} light off.",
    ]
    lock_phrases   = ["Lock the {d}.", "Close the {d}.", "Secure the {d}.",
                      "Can you lock the {d}?", "Please lock the {d}."]
    unlock_phrases = ["Unlock the {d}.", "Open the {d}.", "Open up the {d}.",
                      "Can you unlock the {d}?"]
 
    half = target // 2
 
    # ── lights ─────────────────────────────────────────────────────────
    for i in range(half):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        desired = random.choice(["on", "off"])
        alias   = random.choice(ROOM_ALIASES[r])
 
        # Alternate: even index = match, odd index = mismatch
        is_match = (i % 2 == 0)
        current_state = desired if is_match else ("off" if desired == "on" else "on")
        apply_force(state, {"lights": {r: current_state}}, avail_r, avail_d)
 
        prompt = random.choice(on_phrases if desired == "on" else off_phrases).format(r=alias)
 
        think = (
            f"Checking STATE: {r}={current_state}. "
            f"User wants {desired}. "
        )
        if is_match:
            think += (
                f"Match — STATE already shows {current_state}. "
                f"No tool call needed."
            )
            examples.append(build_ex(
                prompt, [],
                f"The {alias} light is already {desired}.",
                avail_r, avail_d, state,
                user_room=r, think_trace=think,
                category="state_grounding"))
        else:
            think += (
                f"Mismatch — STATE shows {current_state}, user wants {desired}. "
                f"Calling toggle_lights(room={r}, state={desired})."
            )
            examples.append(build_ex(
                prompt,
                [{"name": "toggle_lights", "args": {"room": r, "state": desired}}],
                f"The {alias} light is now {desired}.",
                avail_r, avail_d, state,
                user_room=r, think_trace=think,
                category="state_grounding"))
 
    # ── doors ──────────────────────────────────────────────────────────
    for i in range(target - half):
        d = random.choice(ALL_DOORS)
        avail_r, avail_d = sample_topology(required_doors=[d])
        state = generate_random_state(avail_r, avail_d)
        desired_action = random.choice(["lock", "unlock"])
        desired_state  = "locked" if desired_action == "lock" else "unlocked"
        alias          = random.choice(DOOR_ALIASES[d])
 
        is_match = (i % 2 == 0)
        current_door_state = desired_state if is_match \
            else ("unlocked" if desired_state == "locked" else "locked")
        apply_force(state, {"doors": {d: current_door_state}}, avail_r, avail_d)
 
        prompt = random.choice(
            lock_phrases if desired_action == "lock" else unlock_phrases
        ).format(d=alias)
 
        think = (
            f"Checking STATE: {d}={current_door_state}. "
            f"User wants {desired_state}. "
        )
        if is_match:
            think += (
                f"Match — STATE already shows {current_door_state}. "
                f"No tool call needed."
            )
            examples.append(build_ex(
                prompt, [],
                f"The {alias} is already {desired_state}.",
                avail_r, avail_d, state,
                think_trace=think, category="state_grounding"))
        else:
            think += (
                f"Mismatch — STATE shows {current_door_state}, "
                f"user wants {desired_state}. "
                f"Calling lock_door(door={d}, state={desired_action})."
            )
            examples.append(build_ex(
                prompt,
                [{"name": "lock_door", "args": {"door": d, "state": desired_action}}],
                f"The {alias} is now {desired_state}.",
                avail_r, avail_d, state,
                think_trace=think, category="state_grounding"))
 
    return examples
 
 
# ══════════════════════════════════════════════════════════════════════
# FIX-V13-B  gen_compound_count_enforcement
# ══════════════════════════════════════════════════════════════════════
 
def gen_compound_count_enforcement(target: int = 2_000) -> list:
    """
    FIX-V13-B: Forces model to count sub-actions BEFORE emitting tool calls.
 
    Key design:
    • Think trace always contains the pattern:
        "Total: N tool calls required. Emitting all N."
      This creates a commitment checkpoint before any <|tool_call_start|> block.
    • N ranges 2–4 to cover the truncation failures at higher counts.
    • Sub-action descriptors are enumerated (1), (2), (3) so the model can
      index into them during generation.
    • Partial-satisfaction variants included so the model also learns to count
      only the NEEDED calls when some states already match.
    """
    examples = []
 
    attempts = 0
    while len(examples) < target and attempts < target * 6:
        attempts += 1
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=2)
        state = generate_random_state(avail_r, avail_d)
 
        n_actions = random.choices([2, 3, 4], weights=[35, 45, 20])[0]
        action_types = random.choices(
            ["light", "door", "scene", "thermostat"],
            k=n_actions
        )
 
        calls      = []
        sub_descs  = []
        resp_parts = []
        prompt_parts = []
        used_rooms = set()
        used_doors = set()
        valid = True
 
        for act in action_types:
            if act == "light":
                available = [r for r in avail_r if r not in used_rooms]
                if not available: valid = False; break
                r = random.choice(available); used_rooms.add(r)
                ls  = random.choice(["on", "off"])
                opp = "off" if ls == "on" else "on"
                already = (state["lights"][r]["state"] == ls)
                alias   = random.choice(ROOM_ALIASES[r])
                if not already:
                    apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)
                    calls.append({"name": "toggle_lights",
                                  "args": {"room": r, "state": ls}})
                    sub_descs.append(f"toggle_lights({r}, {ls})")
                    resp_parts.append(f"{alias.title()} light {ls}")
                else:
                    sub_descs.append(f"{r} light already {ls} — skip")
                    resp_parts.append(f"The {alias} light is already {ls}")
                prompt_parts.append(f"turn {ls} the {alias} light")
 
            elif act == "door":
                available = [d for d in avail_d if d not in used_doors]
                if not available: valid = False; break
                d = random.choice(available); used_doors.add(d)
                ds  = random.choice(["lock", "unlock"])
                aw  = "locked" if ds == "lock" else "unlocked"
                opp_aw = "unlocked" if ds == "lock" else "locked"
                already = (state["doors"][d] == aw)
                da = random.choice(DOOR_ALIASES[d])
                if not already:
                    apply_force(state, {"doors": {d: opp_aw}}, avail_r, avail_d)
                    calls.append({"name": "lock_door",
                                  "args": {"door": d, "state": ds}})
                    sub_descs.append(f"lock_door({d}, {ds})")
                    resp_parts.append(f"{da.title()} {aw}")
                else:
                    sub_descs.append(f"{d} door already {aw} — skip")
                    resp_parts.append(f"The {da} is already {aw}")
                prompt_parts.append(f"{ds} the {da}")
 
            elif act == "scene":
                scene = random.choice(SCENES)
                already = (state.get("active_scene") == scene)
                if not already:
                    state["active_scene"] = None
                    calls.append({"name": "set_scene", "args": {"scene": scene}})
                    sub_descs.append(f"set_scene({scene})")
                    resp_parts.append(SCENE_RESP[scene].rstrip("."))
                else:
                    sub_descs.append(f"scene {scene} already active — skip")
                    resp_parts.append(
                        f"The {scene.replace('_',' ').title()} scene is already active")
                prompt_parts.append(f"activate {scene.replace('_', ' ')} mode")
 
            elif act == "thermostat":
                cur = state["thermostat"]["temperature"]
                val = random.randint(MIN_T, MAX_T)
                while val == cur: val = random.randint(MIN_T, MAX_T)
                mode = "cool" if val < cur else "heat"
                calls.append({"name": "set_thermostat",
                              "args": {"temperature": val, "mode": mode}})
                sub_descs.append(f"set_thermostat({val}, {mode})")
                resp_parts.append(f"Thermostat set to {val}°F in {mode} mode")
                prompt_parts.append(f"set thermostat to {val}")
 
        if not valid or len(prompt_parts) < 2:
            continue
 
        n_calls = len(calls)
        n_subs  = len(sub_descs)
 
        numbered = " ".join(
            f"({i+1}) {d}{',' if i < n_subs-1 else '.'}"
            for i, d in enumerate(sub_descs)
        )
 
        think = (
            f"Compound request. Counting sub-actions: {numbered} "
            f"Total: {n_calls} tool call{'s' if n_calls != 1 else ''} required. "
            f"Emitting all {n_calls}."
        )
 
        if len(prompt_parts) == 2:
            prompt = f"{prompt_parts[0].capitalize()} and {prompt_parts[1]}."
        elif len(prompt_parts) == 3:
            prompt = (f"{prompt_parts[0].capitalize()}, "
                      f"{prompt_parts[1]}, and {prompt_parts[2]}.")
        else:
            prompt = (f"{prompt_parts[0].capitalize()}, "
                      + ", ".join(prompt_parts[1:-1])
                      + f", and {prompt_parts[-1]}.")
 
        resp = "  ".join(
            (p + ".") if not p.rstrip().endswith(".") else p
            for p in resp_parts
        )
 
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.3 else ""
 
        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think,
            category="compound_count"))
 
    return examples[:target]
 
 
# ══════════════════════════════════════════════════════════════════════
# FIX-V13-C  gen_them_plurality
# ══════════════════════════════════════════════════════════════════════
 
def gen_them_plurality(target: int = 2_000) -> list:
    examples = []
    
    off_phrases = [
        "Off them.", "Turn them off.", "Kill them.", "Turn them all off.",
        "Off them all.", "Switch them off.", "Kill those lights.",
        "Turn those off.", "Off those.", "Put them all off.", "Off those lights.",
        "Shut them all off.", "Cut them.", "Them off.", "Lights off.",
    ]
    on_phrases = [
        "On them.", "Turn them on.", "On them back.", "Turn them back on.",
        "Put them back on.", "Switch them on.", "On those lights.",
        "Turn those on.", "Bring them back on.", "On those back.",
        "Turn them all on.", "Them on.", "On those.", "Lights back on.",
    ]

    for _ in range(target):
        device_type = random.choice(["light", "door"])
        
        if device_type == "light":
            n_rooms = random.choices([2, 3, 4, 5], weights=[15, 30, 35, 20])[0]
            avail_r, avail_d = sample_topology(min_rooms=max(n_rooms + 1, 3))
            rooms = random.sample(avail_r, min(n_rooms, len(avail_r)))
            if len(rooms) < 2: continue
            state = generate_random_state(avail_r, avail_d)
            prior_state = random.choice(["on", "off"])
            target_state = "off" if prior_state == "on" else "on"
            
            for r in rooms: apply_force(state, {"lights": {r: prior_state}}, avail_r, avail_d)
            call_strs = [f"toggle_lights(room={r}, state={prior_state})" for r in rooms]
            summary = " ".join(f"{ROOM_DISPLAY[r].title()} light turned {'on' if prior_state == 'on' else 'off'}." for r in rooms)
            
            is_reversal = (random.random() < 0.70)
            final_state = target_state if is_reversal else prior_state
            phrase = random.choice(on_phrases if final_state == "on" else off_phrases)
            device_list = ", ".join(ROOM_DISPLAY[r] for r in rooms)
            
            # THE FIX: We evaluate match/mismatch naturally instead of forcing state later
            if is_reversal:
                calls = [{"name": "toggle_lights", "args": {"room": r, "state": final_state}} for r in rooms]
                resp = " ".join(f"{ROOM_DISPLAY[r].title()} light {final_state}." for r in rooms)
                think_eval = f"Current state is {prior_state}, user wants {final_state}. Mismatch — action required. Issuing exactly {len(rooms)} calls, one per device."
            else:
                calls = []
                resp = f"Those lights are already {final_state}."
                think_eval = f"Current state is {prior_state}, user wants {final_state}. Match — no action required."

            n_count = len(rooms)
            dev_name = "lights"
            
        else: # DOOR
            n_doors = random.choices([2, 3, 4, 5], weights=[15, 30, 35, 20])[0]
            avail_r, avail_d = sample_topology(min_doors=max(n_doors + 1, 3))
            doors = random.sample(avail_d, min(n_doors, len(avail_d)))
            if len(doors) < 2: continue
            state = generate_random_state(avail_r, avail_d)
            prior_state = random.choice(["lock", "unlock"])
            target_state = "unlock" if prior_state == "lock" else "lock"
            aw = "locked" if prior_state == "lock" else "unlocked"
            
            for d in doors: apply_force(state, {"doors": {d: aw}}, avail_r, avail_d)
            call_strs = [f"lock_door(door={d}, state={prior_state})" for d in doors]
            summary = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in doors)
            
            is_reversal = (random.random() < 0.70)
            final_state = target_state if is_reversal else prior_state
            door_off_phrases = ["Unlock them.", "Open them.", "Open them all.", "Unlock those doors."]
            door_on_phrases = ["Lock them.", "Close them.", "Secure them all.", "Lock those doors."]
            phrase = random.choice(door_on_phrases if final_state == "lock" else door_off_phrases)
            device_list = ", ".join(DOOR_DISPLAY[d] for d in doors)
            
            if is_reversal:
                calls = [{"name": "lock_door", "args": {"door": d, "state": final_state}} for d in doors]
                taw = "locked" if final_state == "lock" else "unlocked"
                resp = " ".join(f"{DOOR_DISPLAY[d].title()} {taw}." for d in doors)
                think_eval = f"Current state is {aw}, user wants {final_state}. Mismatch — action required. Issuing exactly {len(doors)} calls, one per device."
            else:
                calls = []
                taw = "locked" if final_state == "lock" else "unlocked"
                resp = f"Those doors are already {taw}."
                think_eval = f"Current state is {aw}, user wants {final_state}. Match — no action required."

            n_count = len(doors)
            dev_name = "doors"

        primary_mins = random.randint(1, 4)
        primary_txn = fmt_txn(primary_mins, call_strs, summary)
        dist = build_distractor_log(avail_r, avail_d, n=random.randint(1, 2), start_mins=primary_mins + random.randint(6, 12))
        action_log = primary_txn + "\n" + dist
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        
        think = (
            f"User said '{phrase}'. "
            f"Pronoun 'them'/'those'/'them all' resolves to the first [...] block ({t_label}) "
            f"which logged {n_count} {dev_name}: {device_list}. "
            f"Resolving strictly to the {n_count} logged {dev_name}. "
            f"{think_eval}"
        )

        user_room = random.choice(avail_r + [""])
        examples.append(build_ex(
            phrase, calls, resp, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="them_plurality"))

    return examples
 

def gen_full_topology_bulk_doors(target: int = 1_000) -> list:
    """Forces all 9 doors in every example so the model learns the full count."""
    examples = []
    for _ in range(target):
        avail_r = list(ALL_ROOMS)
        avail_d = list(ALL_DOORS)   # always all 9
        state = generate_random_state(avail_r, avail_d)
        s = random.choice(["lock", "unlock"])
        aw = "locked" if s == "lock" else "unlocked"
        opp = "unlocked" if s == "lock" else "locked"
        action_doors = [d for d in avail_d if state["doors"][d] == opp]
        prompt = random.choice(["Lock all the doors.", "Open all the doors.",
                         "Secure every door.", "Unlock all doors."])
        if not action_doors:
            summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            think = (
                f"User said '{prompt}'. Global door scope. "
                f"Checking ALL 9 connected doors: {summary}. "
                f"Result: ALL 9 doors already {aw}. "
                f"State already matches request. No tool calls needed."
            )
            examples.append(build_ex(prompt, [],
                f"All doors are already {aw}.",
                avail_r, avail_d, state,
                think_trace=think,
                category="full_topology_bulk_doors"))
            continue
        summary = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
        all_names = ", ".join(DOOR_DISPLAY[d] for d in action_doors)
        prompt = random.choice(["Lock all the doors.", "Open all the doors.",
                                 "Secure every door.", "Unlock all doors."])
        think = (
            f"User said '{prompt}'. Global door scope. "
            f"Checking ALL 9 connected doors: {summary}. "
            f"{len(action_doors)} door(s) need to {s} ({all_names}). "
            f"Issuing {len(action_doors)} lock_door calls individually."
        )
        calls = [{"name": "lock_door", "args": {"door": d, "state": s}}
                 for d in action_doors]
        resp = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in action_doors)
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            think_trace=think, category="full_topology_bulk_doors"))
    return examples

def gen_living_room_door_positive(target: int = 400) -> list:
    """Counteracts false unsupported_device for living_room door."""
    examples = []
    for _ in range(target):
        avail_r, avail_d = sample_topology(required_doors=["living_room"])
        state = generate_random_state(avail_r, avail_d)
        s = random.choice(["lock", "unlock"])
        aw = "locked" if s == "lock" else "unlocked"
        apply_force(state, {"doors": {"living_room": "unlocked" if s == "lock" else "locked"}},
                    avail_r, avail_d)
        verb = random.choice(["close", "lock", "open", "unlock"])
        phrase = random.choice(['this ', 'the ', ''])
        prompt = f"{verb.capitalize()} {phrase}door."
        conn_str = ", ".join(avail_d)
        think = (
            f"User Said '{prompt}'. "
            f"'{phrase} door' resolves to current_user_room='living_room'. "
            f"Checking CONNECTED DOORS from system prompt: [{conn_str}]. "
            f"'living_room' IS in this list. "
            f"Calling lock_door(door=living_room, state={s})."
        )
        examples.append(build_ex(prompt,
            [{"name": "lock_door", "args": {"door": "living_room", "state": s}}],
            f"The living room door is now {aw}.",
            avail_r, avail_d, state, user_room="living_room",
            think_trace=think, category="living_room_door_positive"))
    return examples

def gen_compound_direction_stress(target: int = 1_000) -> list:
    """Trains correct on/off parsing inside compound commands."""
    examples = []
    for _ in range(target):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        ls = random.choice(["on", "off"])
        opp = "off" if ls == "on" else "on"
        apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)
        alias = random.choice(ROOM_ALIASES[r])
        t = random.randint(65, 78)
        cur = state["thermostat"]["temperature"]
        while t == cur: t = random.randint(65, 78)
        mode = "cool" if t < cur else "heat"
        prompt = f"{ls.capitalize()} the {alias} light and set temp to {t}."
        think = (
            f"Compound request. (1) User said '{ls} the {alias} light' — "
            f"parsed direction is '{ls}' "
            f"toggle_lights(room={r}, state={ls}). "
            f"(2) set_thermostat(temperature={t}, mode={mode})."
        )
        calls = [
            {"name": "toggle_lights", "args": {"room": r, "state": ls}},
            {"name": "set_thermostat", "args": {"temperature": t, "mode": mode}},
        ]
        resp = f"The {alias} light is now {ls}. Thermostat set to {t}°F in {mode} mode."
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=r, think_trace=think, category="compound_direction_stress"))
    return examples

def gen_action_log_queries(target: int = 1_500) -> list:
    """Trains the model to answer 'what did you just do?' by reading the log."""
    examples = []
    queries = [
        "What did you just do?", "What was the last thing you did?",
        "What was that?", "Did you just do something?", 
        "What action did you just take?", "Tell me what you just changed."
    ]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=2)
        state = generate_random_state(avail_r, avail_d)
        primary_mins = random.randint(1, 3)
        r = random.choice(avail_r)
        action_s = random.choice(["on", "off"])
        summary_text = f"{ROOM_DISPLAY[r].title()} light turned {action_s}."
        primary_txn = fmt_txn(primary_mins, [f"toggle_lights(room={r}, state={action_s})"], summary_text)
        action_log = primary_txn + "\n" + build_distractor_log(avail_r, avail_d, n=1, start_mins=primary_mins + 5)
        
        prompt = random.choice(queries)
        think = (
            f"User asks '{prompt}'. This is a read-only query about RECENT ACTIONS. "
            f"Reading the first [...] block: {summary_text} "
            f"Composing text reply. No tool call needed."
        )
        examples.append(build_ex(
            prompt, [], f"The last thing I did was: {summary_text}", 
            avail_r, avail_d, state, action_log=action_log,
            think_trace=think, category="action_log_queries"
        ))
    return examples


# ══════════════════════════════════════════════════════════════════════
# FIX-V13-D  gen_bulk_plus_local_door
# ══════════════════════════════════════════════════════════════════════
 
def gen_bulk_plus_local_door(target: int = 1_500) -> list:
    examples = []
 
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]
 
    off_on_phrases  = [
        "off the lights that are on", "turn off what's on",
        "kill the lights that are on", "off the on lights",
        "turn off all lights currently on", "switch off the on lights",
    ]
    all_off_phrases = [
        "off all the lights", "turn off all the lights",
        "all lights off", "kill all the lights", "every light off",
    ]
    all_on_phrases  = [
        "on all the lights", "turn on all the lights",
        "all lights on", "every light on", "lights on everywhere",
    ]
    close_phrases   = [
        "close this door", "lock this door", "shut this door",
        "close the door", "lock the door", "secure this door",
    ]
    open_phrases    = [
        "open this door", "unlock this door",
        "open the door", "unlock the door",
    ]
 
    attempts = 0
    while len(examples) < target and attempts < target * 5:
        attempts += 1
        u_room = random.choice(ROOM_DOOR_ROOMS)
        avail_r, avail_d = sample_topology(
            required_rooms=[u_room],
            required_doors=[u_room],
            min_rooms=3)
        state = generate_random_state(avail_r, avail_d)
 
        bulk_type   = random.choice(["off_on", "all_off", "all_on"])
        door_action = random.choice(["lock", "unlock"])
        door_aw     = "locked" if door_action == "lock" else "unlocked"
        door_opp    = "unlocked" if door_action == "lock" else "locked"
        apply_force(state, {"doors": {u_room: door_opp}}, avail_r, avail_d)
        da = random.choice(DOOR_ALIASES[u_room])
 
        calls       = []
        bulk_frag   = ""
        part1_think = ""
        bulk_resp   = ""
 
        if bulk_type == "off_on":
            n_on = random.randint(1, len(avail_r))
            on_rooms  = random.sample(avail_r, n_on)
            off_rooms = [r for r in avail_r if r not in on_rooms]
            for r in on_rooms:
                apply_force(state, {"lights": {r: "on"}}, avail_r, avail_d)
            for r in off_rooms:
                apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
 
            on_names    = ", ".join(ROOM_DISPLAY[r] for r in on_rooms)
            all_summary = ", ".join(
                f"{ROOM_DISPLAY[r]}:{state['lights'][r]['state']}" for r in avail_r
            )
            for r in on_rooms:
                calls.append({"name": "toggle_lights", "args": {"room": r, "state": "off"}})
            bulk_frag = random.choice(off_on_phrases)
            part1_think = (
                f"(1) '{bulk_frag}': "
                f"Checking ALL connected lights: {all_summary}. "
                f"{len(on_rooms)} light(s) currently on ({on_names}). "
                f"Issuing {len(on_rooms)} toggle_lights(off) calls."
            )
            bulk_resp = " ".join(f"{ROOM_DISPLAY[r].title()} light off." for r in on_rooms)
 
        elif bulk_type == "all_off":
            for r2 in avail_r:
                apply_force(state, {"lights": {r2: "on"}}, avail_r, avail_d)
            on_rs   = list(avail_r)
            summary = ", ".join(f"{ROOM_DISPLAY[r2]}:on" for r2 in on_rs)
            for r2 in on_rs:
                calls.append({"name": "toggle_lights", "args": {"room": r2, "state": "off"}})
            bulk_frag   = random.choice(all_off_phrases)
            part1_think = (
                f"(1) '{bulk_frag}': "
                f"Checking ALL connected lights: {summary}. "
                f"All {len(on_rs)} light(s) currently on. "
                f"Issuing {len(on_rs)} toggle_lights(off) calls individually."
            )
            bulk_resp = " ".join(f"{ROOM_DISPLAY[r2].title()} light off." for r2 in on_rs)
 
        else:  # all_on
            for r2 in avail_r:
                apply_force(state, {"lights": {r2: "off"}}, avail_r, avail_d)
            off_rs  = list(avail_r)
            summary = ", ".join(f"{ROOM_DISPLAY[r2]}:off" for r2 in off_rs)
            for r2 in off_rs:
                calls.append({"name": "toggle_lights", "args": {"room": r2, "state": "on"}})
            bulk_frag   = random.choice(all_on_phrases)
            part1_think = (
                f"(1) Light scope — '{bulk_frag}': "
                f"Checking ALL connected lights: {summary}. "
                f"All {len(off_rs)} light(s) currently off. "
                f"Issuing {len(off_rs)} toggle_lights(on) calls individually."
            )
            bulk_resp = " ".join(f"{ROOM_DISPLAY[r2].title()} light on." for r2 in off_rs)
 
        door_phrase = random.choice(close_phrases if door_action == "lock" else open_phrases)
        calls.append({"name": "lock_door", "args": {"door": u_room, "state": door_action}})
 
        n_total = len(calls)
        think = (
            f"Compound request — bulk light command and local door action. "
            f"Evaluating each part separately. "
            f"{part1_think} "
            f"(2) '{door_phrase}': no door explicitly named. "
            f"current_user_room='{u_room}'. Calling lock_door(door={u_room}, state={door_action}). "
            f"Total: {n_total} tool call{'s' if n_total != 1 else ''} required. Emitting all {n_total}."
        )
 
        prompt = random.choice([
            f"{bulk_frag.capitalize()} and {door_phrase}.",
            f"{door_phrase.capitalize()} and {bulk_frag}.",
            f"Can you {bulk_frag} and {door_phrase}?",
            f"Please {bulk_frag} and {door_phrase}.",
        ])
        resp = f"{bulk_resp} {da.title()} {door_aw}."
 
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.3 else ""
 
        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=u_room, action_log=action_log,
            think_trace=think, category="bulk_plus_local_door"))
 
    return examples[:target]



def gen_lock_all_doors_reinforced(target: int = 1_000) -> list:
    """
    Reinforces: 'all doors'/'every door' → check each door's state →
    issue individual lock_door calls only for doors that need changing.
    Same immutable rule as all other devices: never act on a device already
    in the requested state.
    """
    examples = []

    all_lock_phrases = [
        "Lock all the doors.", "Lock all doors.", "Secure all doors.",
        "Lock every door.", "Close all the doors.", "Lock everything.",
        "Close all doors.", "Secure every door.", "Lock all doors please.",
        "Can you lock all the doors?", "Lock them all.",
    ]
    all_unlock_phrases = [
        "Unlock all the doors.", "Open all doors.", "Unlock all doors.",
        "Unlock every door.", "Open all the doors.", "Unlock everything.",
        "Open every door.", "Can you unlock all the doors?",
    ]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_doors=3)
        state     = generate_random_state(avail_r, avail_d)
        s         = random.choice(["lock", "unlock"])
        aw        = "locked"   if s == "lock"   else "unlocked"
        opp_aw    = "unlocked" if s == "lock"   else "locked"
        user_room = random.choice([""] + avail_r)
        prompt    = random.choice(all_lock_phrases if s == "lock" else all_unlock_phrases)
        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""

        action_doors = [d for d in avail_d if state["doors"][d] == opp_aw]
        if not action_doors:
            all_summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            think = (
                f"User said '{prompt}'. "
                f"'All doors'/'every door' — checking every connected door. "
                f"Checking ALL connected door states: {all_summary}. "
                f"Result: ALL doors already {aw}. "
                f"State already matches request. No tool calls needed."
            )
            examples.append(build_ex(prompt, [],
                f"All doors are already {aw}.",
                avail_r, avail_d, state,
                user_room=user_room, action_log=action_log,
                think_trace=think,
                category="lock_all_doors_reinforced"))
            continue

        skipped     = [d for d in avail_d if d not in action_doors]
        all_summary = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
        act_names   = ", ".join(DOOR_DISPLAY[d] for d in action_doors)

        think = (
            f"User said '{prompt}'. "
            f"'All doors'/'every door' means check every connected door and act only "
            f"on those whose state contradicts the request — same rule as all devices. "
            f"Checking ALL connected door states: {all_summary}. "
            f"{len(action_doors)} door(s) currently {opp_aw} ({act_names}) — need to {s}. "
            + (f"{len(skipped)} already {aw} — skipping those. " if skipped else "")
            + f"Issuing {len(action_doors)} individual lock_door(state={s}) call(s)."
        )
        calls = [{"name": "lock_door", "args": {"door": d, "state": s}} for d in action_doors]
        resp  = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in action_doors)

        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="lock_all_doors_reinforced"))

    return examples

def gen_door_incomplete_vs_unsupported(target: int = 800) -> list:
    """
    Targeted fix for the incomplete vs unsupported_device confusion on door commands.
    
    The model confuses:
      A) current_user_room=""  + "Lock the door."  → incomplete  (don't know WHICH door)
      B) current_user_room="hallway" + "Lock the door." + hallway not in avail_d 
                                                       → unsupported_device

    Both are trained elsewhere but in insufficient contrast. This generator
    produces paired examples that make the distinction explicit in the think trace.
    """
    examples = []
    lk_phrases = ["Lock the door.", "Close the door.", "Secure the door.",
                  "Shut the door.", "Lock it.", "Close this door."]
    ul_phrases = ["Unlock the door.", "Open the door.", "Open it up.",
                  "Unlock it.", "Open this door."]

    # A — empty room → incomplete (50%)
    for _ in range(target // 2):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        s = random.choice(["lock", "unlock"])
        prompt = random.choice(lk_phrases if s == "lock" else ul_phrases)
        think = (
            f"User said '{prompt}'. "
            f"current_user_room is empty — lacking room context to determine the target door. "
            f"User intent is clear but the target is ambiguous, marking request as incomplete. "
            f"Calling intent_unclear(incomplete)."
        )
        examples.append(build_ex(prompt,
            [{"name": "intent_unclear", "args": {"reason": "incomplete"}}],
            "Which door would you like me to lock?",
            avail_r, avail_d, state, user_room="",
            think_trace=think, category="door_incomplete_vs_unsupported"))

    # B — room set but door not connected → unsupported_device (50%)
    for _ in range(target // 2):
        avail_r, avail_d = sample_topology()
        state = generate_random_state(avail_r, avail_d)
        rooms_without_door = [r for r in avail_r if r not in avail_d]
        if not rooms_without_door:
            continue
        u_room = random.choice(rooms_without_door)
        s = random.choice(["lock", "unlock"])
        prompt = random.choice(lk_phrases if s == "lock" else ul_phrases)
        conn_str = ", ".join(avail_d)
        think = (
            f"User said '{prompt}'. "
            f"current_user_room='{u_room}' is set. "
            f"'The door' resolves to current_user_room='{u_room}'. "
            f"Checking CONNECTED DOORS: [{conn_str}] '{u_room}' is NOT in the connected doors list. "
            f"Calling intent_unclear(unsupported_device)."
        )
        examples.append(build_ex(prompt,
            [{"name": "intent_unclear", "args": {"reason": "unsupported_device"}}],
            f"There's no door connected in the {ROOM_DISPLAY[u_room]}.",
            avail_r, avail_d, state, user_room=u_room,
            think_trace=think, category="door_incomplete_vs_unsupported"))

    return examples

def gen_self_contradictory_compound(target: int = 2000) -> list:
    """
    Trains two edge cases that fall outside every other generator:
    A) Relative clause is already satisfied ("on the light that is on").
    B) Physically contradictory compound ("on and off the bedroom light").
    """
    examples = []

    # A — already-satisfied relative clause
    for _ in range(target // 2):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state   = generate_random_state(avail_r, avail_d)
        desired = random.choice(["on", "off"])
        apply_force(state, {"lights": {r: desired}}, avail_r, avail_d)
        alias = random.choice(ROOM_ALIASES[r])
        prompt = (
            f"Turn {'on' if desired=='on' else 'off'} "
            f"the {alias} light that is {desired}."
        )
        think = (
            f"User said '{prompt}'. Checking STATE: {r}={desired}. "
            f"Relative clause ('that is {desired}') matches STATE. "
            f"Match — STATE already shows {desired}. No tool call needed."
        )
        examples.append(build_ex(prompt, [],
            f"The {alias} light is already {desired}.",
            avail_r, avail_d, state, user_room=r,
            think_trace=think, category="self_contradictory"))

    # B — physically contradictory compound
    for _ in range(target // 2):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        alias = random.choice(ROOM_ALIASES[r])
        prompt = random.choice([
            f"Turn on and turn off the {alias} light.",
            f"On the {alias} light and off the {alias} light.",
            f"Switch the {alias} light on and off.",
        ])
        think = (
            f"User requested conflicting states (on AND off) for the same {alias} light. "
            f"Cannot satisfy both simultaneously. "
            f"Calling intent_unclear(incomplete)."
        )
        examples.append(build_ex(prompt,
            [{"name": "intent_unclear", "args": {"reason": "incomplete"}}],
            f"I can't turn the {alias} light both on and off — which would you like?",
            avail_r, avail_d, state, user_room=r,
            think_trace=think, category="self_contradictory"))

    return examples



def gen_log_plus_relative_clause(target: int = 600) -> list:
    """
    Trains: RECENT ACTIONS present + relative state clause in the same turn.
    The relative clause must override both current_user_room and the action log.
    Closes the gap where these two mechanisms were only ever trained in isolation.
    """
    examples = []

    phrases_on  = [
        "on the light that is on",  "turn on the one that is on",
        "switch on the light that's currently on",
    ]
    phrases_off = [
        "off the light that is off", "turn off the one that is off",
        "kill the light that is currently off",
    ]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3)
        state    = generate_random_state(avail_r, avail_d)
        target_s = random.choice(["on", "off"])
        action_s = "off" if target_s == "on" else "on"

        # Force exactly ONE light into target_s so clause is unambiguous
        for r in avail_r:
            apply_force(state, {"lights": {r: action_s}}, avail_r, avail_d)
        ref_room = random.choice(avail_r)
        state["lights"][ref_room]["state"] = target_s

        other_rooms = [r for r in avail_r if r != ref_room]
        log_room    = random.choice(other_rooms)
        user_room   = random.choice(other_rooms)

        primary_mins = random.randint(1, 4)
        log_s        = random.choice(["on", "off"])
        action_log   = (
            fmt_txn(primary_mins,
                    [f"toggle_lights(room={log_room}, state={log_s})"],
                    f"{ROOM_DISPLAY[log_room]} light turned {log_s}.")
            + "\n"
            + build_distractor_log(avail_r, avail_d, n=1,
                                   start_mins=primary_mins + random.randint(8, 14))
        )

        prompt    = random.choice(phrases_on if target_s == "on" else phrases_off)
        ref_alias = random.choice(ROOM_ALIASES[ref_room])
        t_label   = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"

        think = (
            f"User said '{prompt}'. This is a relative state clause. "
            f"Relative state clauses resolve directly against STATE. "
            f"Scanning ALL connected light states in STATE: "
            f"only '{ref_room}' has light={target_s}. "
            f"Calling toggle_lights(room={ref_room}, state={action_s})."
        )
        examples.append(build_ex(
            prompt,
            [{"name": "toggle_lights", "args": {"room": ref_room, "state": action_s}}],
            f"The {ref_alias} light is now {action_s}.",
            avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="log_plus_relative_clause"))

    return examples

def gen_washroom_boost() -> list:
    """Washroom/bathroom alias training. No action log."""
    examples = []
    pairs = [
        ("Turn on the washroom light.",  "bathroom", "on",  "off"),
        ("Turn off the washroom light.", "bathroom", "off", "on"),
        ("The washroom light is on.",    "bathroom", "on",  "on"),
    ]
    for prompt, r, desired, cur in pairs:
        for _ in range(10):
            avail_r, avail_d = sample_topology(required_rooms=[r])
            state = generate_random_state(avail_r, avail_d)
            apply_force(state, {"lights": {r: cur}}, avail_r, avail_d)
            if cur == desired:
                examples.append(build_ex(prompt, [],
                    f"The washroom light is already {desired}.",
                    avail_r, avail_d, state,
                    category="already_satisfied", augment=False))
            else:
                examples.append(build_ex(prompt,
                    [{"name": "toggle_lights", "args": {"room": r, "state": desired}}],
                    f"The washroom light is now {desired}.",
                    avail_r, avail_d, state,
                    category="action_required", augment=False))
    return examples

def gen_speaker_explicit_room_multi(target: int = 3_000) -> list:
    examples = []
    tmpls = [
        "Play music in the {r}.", "Start the {r} speaker.",
        "On the {r} speaker.", "Play something in the {r}.",
        "Stop the {r} speaker.", "Pause the {r} speaker.",
        "Turn off the {r} speaker.", "Stop music in the {r}.",
    ]
    action_map = {
        "Play music in the {r}.": "play",
        "Start the {r} speaker.": "play",
        "On the {r} speaker.": "play",
        "Play something in the {r}.": "play",
        "Stop the {r} speaker.": "stop",
        "Pause the {r} speaker.": "pause",
        "Turn off the {r} speaker.": "stop",
        "Stop music in the {r}.": "stop",
    }
    for _ in range(target):
        sp_rooms = random.sample(SPEAKER_ROOMS, random.randint(2, min(3, len(SPEAKER_ROOMS))))
        avail_r, avail_d = sample_topology(required_rooms=sp_rooms)
        state = generate_random_state(avail_r, avail_d)
        r = random.choice(sp_rooms)
        alias = random.choice(ROOM_ALIASES[r])
        tmpl = random.choice(tmpls)
        action = action_map[tmpl]
        prompt = tmpl.format(r=alias)
        if action == "play":
            state["speaker"][r] = "stopped"
        elif action in ("stop", "pause"):
            state["speaker"][r] = "playing"
        # VARY USER ROOM: explicit name wins regardless of where user is
        user_room = random.choice(["", r] + [x for x in avail_r if x != r])  # ← FIX
        think = (
            f"User explicitly named the '{alias}' ({r}) speaker. "
            f"Target room is '{r}'. "
            f"Calling control_speaker(room={r}, action={action})."
        )
        resp_map = {
            "play": f"Playing music on the {alias} speaker.",
            "stop": f"Stopped the music on the {alias} speaker.",
            "pause": f"Paused the {alias} speaker.",
        }
        examples.append(build_ex(prompt,
            [{"name": "control_speaker", "args": {"room": r, "action": action}}],
            resp_map[action], avail_r, avail_d, state,
            user_room=user_room,  # ← FIX: was always ""
            think_trace=think, category="speaker_explicit_room_multi"))
    return examples




def gen_compound_relative_plus_gadget(target: int = 1_000) -> list:
    """
    Relative state clause + gadget action in same compound command.
    'Off the light that is on and play some music.'
    Relative clause overrides current_user_room for the light.
    Speaker resolved via Rule 1 (single) or Rule 2 (user in speaker room).
    """
    examples = []

    light_phrases = [
        "off the light that is on", "turn off the one that is on",
        "kill the light that's on", "switch off the light that is currently on",
        "turn off the light that's currently on",
    ]
    music_phrases = [
        "Play {m}.", "Can you play {m}?", "Put on {m}.", "I want to hear {m}.","Put for me {m}.",
        "let's Listen to {m}.", "Queue up {m} for me.", "Play me {m}.", "Queue up {m} for me.",
        "play {m} for me.",
    ]

    attempts = 0
    while len(examples) < target and attempts < target * 4:
        attempts += 1
        avail_r, avail_d = sample_topology(min_rooms=3)
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not sp_rooms:
            continue

        state = generate_random_state(avail_r, avail_d)

        # Force exactly one light ON so relative clause is unambiguous
        for r in avail_r:
            state["lights"][r]["state"] = "off"
        on_room = random.choice(avail_r)
        state["lights"][on_room]["state"] = "on"

        # Speaker resolution — Rule 1 or Rule 2 only (keeps logic clean)
        media = random.choice(LOCAL_MUSIC)
        sp_str = ", ".join(sp_rooms)
        if len(sp_rooms) == 1:
            music_target = sp_rooms[0]
            state["speaker"][music_target] = "stopped"
            other_rooms = [r for r in avail_r if r != on_room]
            user_room = random.choice(other_rooms + [""]) if other_rooms else ""
            m_think = (
                f"Checking CONNECTED SPEAKERS: [{sp_str}]. "
                f"Exactly one speaker connected. "
                f"Resolving to {music_target}. "
                f"Calling control_speaker(room={music_target}, action=play, media='{media}')."
            )
        else:
            music_target = random.choice(sp_rooms)
            user_room    = music_target
            for r in sp_rooms:
                state["speaker"][r] = "stopped"
            m_think = (
                f"Checking CONNECTED SPEAKERS: [{sp_str}]. Multiple speakers connected. "
                f"current_user_room='{user_room}' has a speaker. "
                f"Resolving to {music_target}. "
                f"Calling control_speaker(room={music_target}, action=play, media='{media}')."
            )
        l_prompt    = random.choice(light_phrases)
        m_prompt    = random.choice(music_phrases).format(m=media)
        on_alias    = random.choice(ROOM_ALIASES[on_room])
        music_alias = random.choice(ROOM_ALIASES[music_target])

        prompt = random.choice([
            f"{l_prompt.capitalize()} and {m_prompt}.",
            f"{m_prompt.capitalize()} and {l_prompt}.",
            f"Can you {l_prompt} and {m_prompt}?",
            f"Please {m_prompt} and {l_prompt}",
        ])

        calls = [
            {"name": "toggle_lights",   "args": {"room": on_room,     "state": "off"}},
            {"name": "control_speaker", "args": {"room": music_target, "action": "play", "media": media}},
        ]
        resp = (
            f"The {on_alias} light is now off. "
            f"Playing music on the {music_alias} speaker."
        )
        think = (
            f"Compound request. Counting sub-actions: "
            f"(1) '{l_prompt}' — relative state clause resolves via STATE. "
            f"Checking STATE: only '{on_room}' has light=on. "
            f"toggle_lights(room={on_room}, state=off). "
            f"(2) '{m_prompt}' — {m_think} "
            f"Total: 2 tool calls required. Emitting all 2."
        )

        action_log = build_distractor_log(avail_r, avail_d, n=1) \
            if random.random() < 0.4 else ""

        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="compound_relative_plus_gadget"
        ))

    return examples


def gen_compound_multi_gadget_explicit(target: int = 1_500) -> list:
    """
    Two different gadget types, each with an explicitly named room.
    'Turn on the bedroom TV and play jazz on the kitchen speaker.'
    Explicit room names override all resolution rules.
    Covers Fan+TV, Fan+Speaker, TV+Speaker combinations.
    """
    examples = []

    attempts = 0
    while len(examples) < target and attempts < target * 5:
        attempts += 1
        avail_r, avail_d = sample_topology(min_rooms=4)
        state = generate_random_state(avail_r, avail_d)

        tv_rooms  = [r for r in avail_r if r in TV_ROOMS]
        fan_rooms = [r for r in avail_r if r in FAN_ROOMS]
        spk_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]

        gadget_pool = []
        if tv_rooms:  gadget_pool.append(("tv",      tv_rooms))
        if fan_rooms: gadget_pool.append(("fan",     fan_rooms))
        if spk_rooms: gadget_pool.append(("speaker", spk_rooms))

        if len(gadget_pool) < 2: continue

        g1_type, g1_rooms = random.choice(gadget_pool)
        remaining = [(t, rs) for t, rs in gadget_pool if t != g1_type]
        if not remaining: continue
        g2_type, g2_rooms = random.choice(remaining)

        r1 = random.choice(g1_rooms)
        r2_opts = [r for r in g2_rooms if r != r1]
        if not r2_opts: continue
        r2 = random.choice(r2_opts)

        a1 = random.choice(ROOM_ALIASES[r1])
        a2 = random.choice(ROOM_ALIASES[r2])

        calls, prompt_parts, resp_parts, think_parts = [], [], [], []

        for gt, rm, al in [(g1_type, r1, a1), (g2_type, r2, a2)]:
            if gt == "tv":
                st = random.choice(["on", "off"])
                state["tv"][rm] = "off" if st == "on" else "on"
                calls.append({"name": "control_tv", "args": {"room": rm, "state": st}})
                prompt_parts.append(f"turn {st} the {al} TV")
                resp_parts.append(f"the {al.title()} TV is now {st}")
                think_parts.append(f"'{al} TV' explicitly named → control_tv(room={rm}, state={st})")
            elif gt == "fan":
                st = random.choice(["on", "off"])
                state["fan"][rm]["state"] = "off" if st == "on" else "on"
                calls.append({"name": "control_fan", "args": {"room": rm, "state": st}})
                prompt_parts.append(f"turn {st} the {al} fan")
                resp_parts.append(f"the {al.title()} fan is now {st}")
                think_parts.append(f"'{al} fan' explicitly named → control_fan(room={rm}, state={st})")
            else:
                st = random.choice(["play", "stop"])
                state["speaker"][rm] = "stopped" if st == "play" else "playing"
                calls.append({"name": "control_speaker", "args": {"room": rm, "action": st}})
                vp = "play music on" if st == "play" else "stop"
                rp = "playing music on" if st == "play" else "stopped"
                prompt_parts.append(f"{vp} the {al} speaker")
                resp_parts.append(f"{rp} the {al} speaker")
                think_parts.append(f"'{al} speaker' explicitly named → control_speaker(room={rm}, action={st})")

        prompt = random.choice([
            f"{prompt_parts[0].capitalize()} and {prompt_parts[1]}.",
            f"Can you {prompt_parts[0]} and {prompt_parts[1]}?",
            f"Please {prompt_parts[0]} and also {prompt_parts[1]}.",
        ])
        resp = f"{resp_parts[0].capitalize()} and {resp_parts[1]}."

        user_room = random.choice(["", r1, r2, random.choice(avail_r)])

        think = (
            f"Compound explicit gadget request. Counting sub-actions: "
            f"(1) {think_parts[0]}. (2) {think_parts[1]}. "
            f"Total: 2 tool calls required. Emitting all 2."
        )

        action_log = build_distractor_log(avail_r, avail_d, n=1) if random.random() < 0.3 else ""

        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="compound_multi_gadget_explicit"
        ))

    return examples


def gen_partial_execution_ambiguous_local(target: int = 2_000) -> list:
    examples = []

    implicit_light_phrases = [
        "turn on this light", "turn off the light", "on the light",
        "off this light", "switch on the light", "turn this light on",
        "kill the light", "on this light",
    ]
    implicit_door_phrases = [
        "close this door", "lock the door", "open this door",
        "unlock the door", "shut this door", "close the door",
        "lock this door", "open up the door",
    ]
    implicit_fan_phrases = [                                   # FIX-G new
        "turn on the fan", "on the fan", "turn off the fan",
        "off the fan", "switch on the fan", "fan on",
    ]

    attempts = 0
    while len(examples) < target and attempts < target * 5:
        attempts += 1
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=3)
        state = generate_random_state(avail_r, avail_d)
        user_room = ""

        # ── Explicit action (always succeeds — room is named) ──
        explicit_type = random.choice(["light", "tv", "speaker"])

        if explicit_type == "light":
            r_exp = random.choice(avail_r)
            a_exp = random.choice(ROOM_ALIASES[r_exp])
            s1    = random.choice(["on", "off"])
            apply_force(state, {"lights": {r_exp: "off" if s1 == "on" else "on"}},
                        avail_r, avail_d)
            calls_exp  = [{"name": "toggle_lights",
                           "args": {"room": r_exp, "state": s1}}]
            prompt_exp = f"turn {s1} the {a_exp} light"
            resp_exp   = f"the {a_exp.title()} light is now {s1}"
            think_exp  = (
                f"(1) '{a_exp} light' is explicitly named — "
                f"toggle_lights(room={r_exp}, state={s1})."
            )

        elif explicit_type == "tv":
            tv_rooms = [r for r in avail_r if r in TV_ROOMS]
            if not tv_rooms: continue
            r_exp = random.choice(tv_rooms)
            a_exp = random.choice(ROOM_ALIASES[r_exp])
            s1    = random.choice(["on", "off"])
            state["tv"][r_exp] = "off" if s1 == "on" else "on"
            calls_exp  = [{"name": "control_tv",
                           "args": {"room": r_exp, "state": s1}}]
            prompt_exp = f"turn {s1} the {a_exp} TV"
            resp_exp   = f"the {a_exp.title()} TV is now {s1}"
            think_exp  = (
                f"(1) '{a_exp} TV' is explicitly named → "
                f"control_tv(room={r_exp}, state={s1})."
            )

        else:  # speaker
            spk_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
            if not spk_rooms: continue
            r_exp = random.choice(spk_rooms)
            a_exp = random.choice(ROOM_ALIASES[r_exp])
            s1    = random.choice(["play", "stop"])
            state["speaker"][r_exp] = "stopped" if s1 == "play" else "playing"
            calls_exp  = [{"name": "control_speaker",
                           "args": {"room": r_exp, "action": s1}}]
            vp         = "play music on" if s1 == "play" else "stop"
            rp         = "started playing music on" if s1 == "play" else "stopped"
            prompt_exp = f"{vp} the {a_exp} speaker"
            resp_exp   = f"{rp} the {a_exp} speaker"
            think_exp  = (
                f"(1) '{a_exp} speaker' is explicitly named → "
                f"control_speaker(room={r_exp}, action={s1})."
            )

        # ── Implicit action (always incomplete — no room context) ──
        # FIX-G: add "fan" when multiple fans are available
        fan_rooms_avail = [r for r in avail_r if r in FAN_ROOMS]
        implicit_options = ["light", "door"]
        if len(fan_rooms_avail) >= 2:
            implicit_options.append("fan")
        implicit_type = random.choice(implicit_options)

        if implicit_type == "light":
            prompt_imp  = random.choice(implicit_light_phrases)
            clarify_msg = "which room's light did you mean"
            think_imp   = (
                f"(2) '{prompt_imp}' — 'the/this light' with no room name. "
                f"current_user_room is empty — cannot determine target. "
                f"intent_unclear(incomplete)."
            )

        elif implicit_type == "door":
            prompt_imp  = random.choice(implicit_door_phrases)
            clarify_msg = "which door did you mean"
            think_imp   = (
                f"(2) '{prompt_imp}' — 'the/this door' with no room name. "
                f"current_user_room is empty — cannot determine target. "
                f"intent_unclear(incomplete)."
            )

        else:  # fan  FIX-G new branch
            desired_fan = random.choice(["on", "off"])
            opp_fan     = "off" if desired_fan == "on" else "on"
            # Force all fans to the eligible state so Rule 4 fires
            for r in fan_rooms_avail:
                state["fan"][r]["state"] = opp_fan
            fan_verb    = "on" if desired_fan == "on" else "off"
            prompt_imp  = random.choice(
                [p for p in implicit_fan_phrases if fan_verb in p]
                or implicit_fan_phrases)
            clarify_msg = "which fan did you mean"
            conn_str    = ", ".join(ROOM_DISPLAY[r] for r in fan_rooms_avail)
            think_imp   = (
                f"(2) '{prompt_imp}' — 'the fan' with no room name. "
                f"Multiple fans connected ({conn_str}), all in eligible state, "
                f"current_user_room is empty. "
                f"Rule 4 applies → intent_unclear(incomplete)."
            )

        calls = calls_exp + [{"name": "intent_unclear",
                               "args": {"reason": "incomplete"}}]

        prompt = random.choice([
            f"{prompt_exp.capitalize()} and {prompt_imp}.",
            f"{prompt_imp.capitalize()} and {prompt_exp}.",
            f"Can you {prompt_exp} and {prompt_imp}?",
        ])
        resp = f"{resp_exp.capitalize()}. However, {clarify_msg}?"

        think = (
            f"Compound request with partial ambiguity. Counting sub-actions: "
            f"{think_exp} "
            f"{think_imp} "
            f"Total: 2 tool calls required. Emitting all 2."
        )

        action_log = (build_distractor_log(avail_r, avail_d, n=1)
                      if random.random() < 0.4 else "")

        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="partial_execution_ambiguous_local"
        ))

    return examples

# ══════════════════════════════════════════════════════════════════════
# SFT ADDITION 1: Bulk Already-Satisfied — the critical gap
# ══════════════════════════════════════════════════════════════════════

def gen_bulk_already_satisfied(target: int = 2_000) -> list:
    """
    Trains: bulk scope command where ALL devices in scope are already
    in the requested state → 0 tool calls for that scope.
    
    Specifically closes Turn 165 pattern:
    'off the lights that are on' when ALL lights are already off.
    
    Also trains the compound variant:
    'off lights that are on AND close doors' where lights all off
    → 0 light calls + N door calls (not N+6 calls).
    """
    examples = []
    
    LIGHTS_ALREADY_OFF_PHRASES = [
        "Turn off the lights that are on.",
        "Off the lights that are on.",
        "Kill the lights that are on.",
        "Turn off all the lights.",
        "All lights off.",
        "Off all the lights.",
        "Lights off please.",
        "Switch off any lights that are on.",
    ]
    LIGHTS_ALREADY_ON_PHRASES = [
        "Turn on all the lights.",
        "All lights on.",
        "On all the lights.",
        "Switch on any lights that are off.",
        "Lights on everywhere.",
    ]
    DOORS_ALREADY_LOCKED_PHRASES = [
        "Close all the doors.", "Lock all the doors.",
        "Lock any doors that are open.", "Secure every door.",
    ]
    DOORS_ALREADY_UNLOCKED_PHRASES = [
        "Open all the doors.", "Unlock all the doors.",
        "Open any doors that are locked.",
    ]
    
    # ── Part 1: Pure bulk already satisfied (single scope) ───────────
    for _ in range(target // 2):
        choice = random.choice(["lights_off", "lights_on", "doors_locked", "doors_unlocked"])
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)
        
        if choice == "lights_off":
            # Force ALL lights already off
            for r in avail_r:
                apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
            prompt = random.choice(LIGHTS_ALREADY_OFF_PHRASES)
            all_summary = ", ".join(
                f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            think = (
                f"User said '{prompt}'. Global light scope. "
                f"Checking ALL connected lights: {all_summary}. "
                f"Result: 0 lights currently on — NONE need to be turned off. "
                f"State already matches request for all lights. No tool calls needed."
            )
            resp = "All lights are already off."
            examples.append(build_ex(prompt, [], resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

        elif choice == "lights_on":
            for r in avail_r:
                apply_force(state, {"lights": {r: "on"}}, avail_r, avail_d)
            prompt = random.choice(LIGHTS_ALREADY_ON_PHRASES)
            all_summary = ", ".join(
                f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            think = (
                f"User said '{prompt}'. Global light scope. "
                f"Checking ALL connected lights: {all_summary}. "
                f"Result: ALL lights are already on — NONE need to be turned on. "
                f"State already matches request for all lights. No tool calls needed."
            )
            resp = "All lights are already on."
            examples.append(build_ex(prompt, [], resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

        elif choice == "doors_locked":
            for d in avail_d:
                apply_force(state, {"doors": {d: "locked"}}, avail_r, avail_d)
            prompt = random.choice(DOORS_ALREADY_LOCKED_PHRASES)
            all_summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            think = (
                f"User said '{prompt}'. Global door scope. "
                f"Checking ALL connected doors: {all_summary}. "
                f"Result: ALL doors already locked — NONE need to be locked. "
                f"State already matches request for all doors. No tool calls needed."
            )
            resp = "All doors are already locked."
            examples.append(build_ex(prompt, [], resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

        else:  # doors_unlocked
            for d in avail_d:
                apply_force(state, {"doors": {d: "unlocked"}}, avail_r, avail_d)
            prompt = random.choice(DOORS_ALREADY_UNLOCKED_PHRASES)
            all_summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            think = (
                f"User said '{prompt}'. Global door scope. "
                f"Checking ALL connected doors: {all_summary}. "
                f"Result: ALL doors already unlocked. No tool calls needed."
            )
            resp = "All doors are already unlocked."
            examples.append(build_ex(prompt, [], resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

    # ── Part 2: Compound — one scope already satisfied, other needs work ─
    # THIS IS THE EXACT PATTERN FROM TURN 165
    for _ in range(target // 2):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)
        
        combo = random.choice([
            "lights_off_doors_need_lock",
            "lights_off_doors_need_unlock", 
            "lights_need_off_doors_locked",
            "lights_on_doors_need_lock",
        ])
        
        if combo == "lights_off_doors_need_lock":
            # Lights all off, doors need locking — the EXACT Turn 165 scenario
            for r in avail_r:
                apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
            # Some doors unlocked
            n_unlock = random.randint(1, len(avail_d))
            unlock_doors = random.sample(avail_d, n_unlock)
            for d in avail_d:
                apply_force(state, 
                    {"doors": {d: "unlocked" if d in unlock_doors else "locked"}},
                    avail_r, avail_d)
            
            prompts_pool = [
                "Off the lights that are on and close all the doors.",
                "Turn off any lights that are on and lock the doors.",
                "Kill the lights that are on and close everything.",
                "Lights off and lock all doors.",
                "Off all lights that are on and close all the doors.",
            ]
            prompt = random.choice(prompts_pool)
            
            l_summary = ", ".join(
                f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            d_summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            unlock_names = ", ".join(DOOR_DISPLAY[d] for d in unlock_doors)
            
            calls = [{"name": "lock_door",
                      "args": {"door": d, "state": "lock"}}
                     for d in unlock_doors]
            resp = ("All lights are already off. "
                    + " ".join(f"{DOOR_DISPLAY[d].title()} locked."
                               for d in unlock_doors))
            
            think = (
                f"Compound request. Evaluating each scope separately. "
                f"(1) Checking ALL connected lights: {l_summary}. "
                f"Result: 0 lights currently on — ALL already off. No calls needed. "
                f"(2) Checking ALL connected doors: {d_summary}. "
                f"Result: {n_unlock} door(s) unlocked ({unlock_names}). "
                f"Issuing {n_unlock} lock_door(state=lock) calls."
                f"Total: {n_unlock} tool call(s) required (0 from lights scope)."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

        elif combo == "lights_need_off_doors_locked":
            # Lights need turning off, doors all already locked
            n_on = random.randint(1, len(avail_r))
            on_rooms = random.sample(avail_r, n_on)
            for r in avail_r:
                apply_force(state,
                    {"lights": {r: "on" if r in on_rooms else "off"}},
                    avail_r, avail_d)
            for d in avail_d:
                apply_force(state, {"doors": {d: "locked"}}, avail_r, avail_d)
            
            prompt = random.choice([
                "Turn off the lights that are on and close all the doors.",
                "Off the lights and lock all the doors.",
                "Kill the lights that are on and secure every door.",
            ])
            
            l_summary = ", ".join(
                f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            d_summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            on_names = ", ".join(ROOM_DISPLAY[r] for r in on_rooms)
            
            calls = [{"name": "toggle_lights",
                      "args": {"room": r, "state": "off"}}
                     for r in on_rooms]
            resp = (" ".join(f"{ROOM_DISPLAY[r].title()} light off."
                             for r in on_rooms)
                    + " All doors are already locked.")
            
            think = (
                f"Compound request. Evaluating each scope separately. "
                f"(1) Lights scope: Checking ALL lights: {l_summary}. "
                f"Result: {n_on} light(s) currently on ({on_names}). "
                f"Issuing {n_on} toggle_lights(off) call(s). "
                f"(2) Door scope: Checking ALL doors: {d_summary}. "
                f"Result: ALL doors already locked. "
                f"SKIP — no door calls needed. "
                f"Total: {n_on} tool call(s) (0 from door scope)."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

        elif combo == "lights_off_doors_need_unlock":
            for r in avail_r:
                apply_force(state, {"lights": {r: "off"}}, avail_r, avail_d)
            n_lock = random.randint(1, len(avail_d))
            lock_doors = random.sample(avail_d, n_lock)
            for d in avail_d:
                apply_force(state,
                    {"doors": {d: "locked" if d in lock_doors else "unlocked"}},
                    avail_r, avail_d)
            prompt = "Off the lights and open all the doors."
            calls = [{"name": "lock_door",
                      "args": {"door": d, "state": "unlock"}}
                     for d in lock_doors]
            lock_names = ", ".join(DOOR_DISPLAY[d] for d in lock_doors)
            l_summary = ", ".join(
                f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            d_summary = ", ".join(
                f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            resp = ("All lights are already off. "
                    + " ".join(f"{DOOR_DISPLAY[d].title()} unlocked."
                               for d in lock_doors))
            think = (
                f"Compound. (1) Lights: {l_summary}. All off — SKIP. "
                f"(2) Doors: {d_summary}. {n_lock} locked ({lock_names}). "
                f"Issuing {n_lock} lock_door(unlock) calls."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

        else:  # lights_on_doors_need_lock
            for r in avail_r:
                apply_force(state, {"lights": {r: "on"}}, avail_r, avail_d)
            n_unlock = random.randint(1, len(avail_d))
            unlock_doors = random.sample(avail_d, n_unlock)
            for d in avail_d:
                apply_force(state,
                    {"doors": {d: "unlocked" if d in unlock_doors else "locked"}},
                    avail_r, avail_d)
            prompt = "On the lights and close all the doors."
            calls = [{"name": "lock_door",
                      "args": {"door": d, "state": "lock"}}
                     for d in unlock_doors]
            unlock_names = ", ".join(DOOR_DISPLAY[d] for d in unlock_doors)
            resp = ("All lights are already on. "
                    + " ".join(f"{DOOR_DISPLAY[d].title()} locked."
                               for d in unlock_doors))
            l_summary = ", ".join(
                f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            think = (
                f"Compound. (1) Lights: {l_summary}. All on — SKIP. "
                f"(2) Doors: {n_unlock} unlocked ({unlock_names}). "
                f"Issuing {n_unlock} lock_door(lock) calls."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                think_trace=think, category="bulk_already_satisfied"))

    return examples


# ══════════════════════════════════════════════════════════════════════
# SFT ADDITION 2: "Off Everything" Universal Scope Command
# ══════════════════════════════════════════════════════════════════════

def gen_off_everything(target: int = 1_500) -> list:
    """
    Trains: 'off everything', 'shut everything down', 'close everywhere'
    Universal scope — checks ALL device types, acts only on those that
    need action, skips those already in the target state.
    """
    examples = []
    
    EVERYTHING_OFF_PHRASES = [
        "Off everything.", "Turn everything off.", "Shut everything down.",
        "Everything off please.", "Kill everything.", "Turn off all devices.",
        "Power everything down.", "Switch everything off.", "Turn it all off.",
        "Off everything in the house.", "Shut it all down.",
    ]
    CLOSE_EVERYWHERE_PHRASES = [
        "Close everywhere.", "Close everything up.", "Lock everything up.",
        "Secure everything.", "Lock everything.", "Close all doors and turn off everything.",
        "Everything locked and off.", "Shut and lock everything.",
    ]
    
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)
        
        is_close_phrase = random.random() < 0.4
        prompt = random.choice(CLOSE_EVERYWHERE_PHRASES if is_close_phrase
                               else EVERYTHING_OFF_PHRASES)
        
        calls = []
        resp_parts = []
        scope_results = []
        
        # ── Lights ────────────────────────────────────────────────────────
        lights_on = [r for r in avail_r if state["lights"][r]["state"] == "on"]
        l_summary = ", ".join(f"{r}:{state['lights'][r]['state']}" for r in avail_r)
        
        for r in lights_on:
            calls.append({"name": "toggle_lights", "args": {"room": r, "state": "off"}})
            resp_parts.append(f"{ROOM_DISPLAY[r].title()} light off.")
        
        scope_results = []
        scope_results.append(
            f"Lights: {len(lights_on)} on, issuing {len(lights_on)} toggle_lights(off) calls."
            if lights_on else f"Lights: {l_summary}. All already off.")
        
        # ── Doors ─────────────────────────────────────────────────────────
        unlocked_doors = [d for d in avail_d if state["doors"][d] == "unlocked"]
        d_summary = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
        
        for d in unlocked_doors:
            calls.append({"name": "lock_door", "args": {"door": d, "state": "lock"}})
            resp_parts.append(f"{DOOR_DISPLAY[d].title()} locked.")
        
        scope_results.append(
            f"Doors: {len(unlocked_doors)} unlocked, issuing {len(unlocked_doors)} lock_door(lock) calls."
            if unlocked_doors else f"Doors: {d_summary}. All already locked.")
        
        # ── TV ────────────────────────────────────────────────────────────
        tv_d   = state.get("tv", {})
        tv_ss  = ", ".join(f"{r}:{tv_d[r]}" for r in sorted(tv_d)) if tv_d else "none"
        tv_on  = [r for r in tv_d if tv_d[r] == "on"]
        
        for r in tv_on:
            calls.append({"name": "control_tv", "args": {"room": r, "state": "off"}})
            resp_parts.append(f"{ROOM_DISPLAY[r].title()} TV off.")
        
        scope_results.append(
            f"TV: {len(tv_on)} on, issuing {len(tv_on)} control_tv(off) calls."
            if tv_on else f"TV: {tv_ss}. All already off.")
        
        # ── Speaker ───────────────────────────────────────────────────────
        sp_d     = state.get("speaker", {})
        sp_ss    = ", ".join(f"{r}:{sp_d[r]}" for r in sorted(sp_d)) if sp_d else "none"
        sp_active = [r for r in sp_d if sp_d[r] in ("playing", "paused")]
        
        for r in sp_active:
            calls.append({"name": "control_speaker", "args": {"room": r, "action": "stop"}})
            resp_parts.append(f"{ROOM_DISPLAY[r].title()} speaker stopped.")
        
        scope_results.append(
            f"Speaker: {len(sp_active)} active, issuing {len(sp_active)} control_speaker(stop) calls."
            if sp_active else f"Speaker: {sp_ss}. All already stopped.")
        
        # ── Fan ───────────────────────────────────────────────────────────
        fan_d  = state.get("fan", {})
        fan_ss = ", ".join(f"{r}:{fan_d[r]['state']}" for r in sorted(fan_d)) if fan_d else "none"
        fans_on = [r for r in fan_d if fan_d[r]["state"] == "on"]
        
        for r in fans_on:
            calls.append({"name": "control_fan", "args": {"room": r, "state": "off"}})
            resp_parts.append(f"{ROOM_DISPLAY[r].title()} fan off.")
        
        scope_results.append(
            f"Fan: {len(fans_on)} on, issuing {len(fans_on)} control_fan(off) calls."
            if fans_on else f"Fan: {fan_ss}. All already off.")
        
        
        if not calls:
            resp = "Everything is already off and locked."
        else:
            resp = " ".join(resp_parts)
        
        n_total = len(calls)
        think = (
            f"User said '{prompt}'. Universal scope — check ALL device types. "
            + " ".join(scope_results)
            + f" Total: {n_total} tool call(s) required."
        )
        
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            think_trace=think, category="off_everything"))
    
    return examples


# ══════════════════════════════════════════════════════════════════════
# SFT ADDITION 3: Wrong Tool Type for Log Devices (Turn 146 pattern)
# ══════════════════════════════════════════════════════════════════════

def gen_pronoun_correct_tool_type(target: int = 1_000) -> list:
    """
    Specifically trains: pronoun 'them/it' on a DOOR log entry
    must use lock_door, not toggle_lights.
    
    Closes Turn 146: 'open them' after door-lock block →
    model called toggle_lights for doors.
    
    The think trace must extract BOTH the room AND the tool type
    from the action log entry.
    """
    examples = []
    
    open_phrases = [
        "Open them.", "Unlock them.", "Open them all.", "Undo that.",
        "Reverse that.", "Open those doors.", "Unlock those.",
    ]
    close_phrases = [
        "Close them.", "Lock them.", "Lock them all.", "Undo that.",
        "Close those.", "Lock those doors.", "Secure them.",
    ]
    
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=4)
        state = generate_random_state(avail_r, avail_d)
        
        # First block: door operations
        n_doors = random.randint(2, min(6, len(avail_d)))
        log_doors = random.sample(avail_d, n_doors)
        log_s = random.choice(["lock", "unlock"])
        log_aw = "locked" if log_s == "lock" else "unlocked"
        rev_s = "unlock" if log_s == "lock" else "lock"
        rev_aw = "unlocked" if log_s == "lock" else "locked"
        
        # Force state to match what was logged
        for d in log_doors:
            apply_force(state, {"doors": {d: log_aw}}, avail_r, avail_d)
        
        call_strs = [f"lock_door(door={d}, state={log_s})" for d in log_doors]
        summary = " ".join(f"{DOOR_DISPLAY[d].title()} {log_aw}." for d in log_doors)
        primary_mins = random.randint(1, 5)
        primary_txn = fmt_txn(primary_mins, call_strs, summary)
        action_log = primary_txn + "\n" + build_distractor_log(
            avail_r, avail_d, n=1, start_mins=primary_mins + 5)
        
        prompt = random.choice(open_phrases if log_s == "lock" else close_phrases)
        
        calls = [{"name": "lock_door",
                  "args": {"door": d, "state": rev_s}}
                 for d in log_doors]
        resp = " ".join(f"{DOOR_DISPLAY[d].title()} {rev_aw}." for d in log_doors)
        
        door_names = ", ".join(DOOR_DISPLAY[d] for d in log_doors)
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        
        think = (
            f"User said '{prompt}'. "
            f"Pronoun 'them'/'those' → first [...] block ({t_label}). "
            f"First block: {n_doors} doors ({door_names}) were {log_aw}. "
            f"Reversing to {rev_s}. "
            f"Issuing {n_doors} lock_door(state={rev_s}) calls."
        )
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think,
            category="pronoun_correct_tool_type"))
    
    return examples




def gen_mixed_device_block_pronoun_undo(target: int = 2_500) -> list:
    examples = []
    undo_phrases = [
        "On them back.", "Undo that.", "Revert that.", "Reverse that.",
        "Put them back.", "Take that back.", "Undo all of that.",
        "Undo everything.", "Take all that back.", "Reverse all of that.",
        "Put everything back.", "Un-do all that.",
    ]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)

        n_lights = random.randint(2, min(4, len(avail_r)))
        n_doors  = random.randint(1, min(3, len(avail_d)))
        log_lights = random.sample(avail_r, n_lights)
        log_doors  = random.sample(avail_d, n_doors)

        light_s   = random.choice(["on", "off"])
        door_s    = random.choice(["lock", "unlock"])
        light_rev = "off" if light_s == "on" else "on"
        door_rev  = "unlock" if door_s == "lock" else "lock"
        light_aw  = "locked" if door_s == "lock" else "unlocked"
        door_rev_aw = "unlocked" if door_s == "lock" else "locked"

        for r in log_lights:
            apply_force(state, {"lights": {r: light_s}}, avail_r, avail_d)
        for d in log_doors:
            apply_force(state, {"doors": {d: light_aw}}, avail_r, avail_d)

        call_strs = (
            [f"toggle_lights(room={r}, state={light_s})" for r in log_lights] +
            [f"lock_door(door={d}, state={door_s})" for d in log_doors]
        )
        light_sum = " ".join(
            f"{ROOM_DISPLAY[r].title()} light {light_s}." for r in log_lights)
        door_sum = " ".join(
            f"{DOOR_DISPLAY[d].title()} {light_aw}." for d in log_doors)
        summary = f"{light_sum} {door_sum}"

        primary_mins = random.randint(1, 3)
        primary_txn  = fmt_txn(primary_mins, call_strs, summary)
        action_log   = primary_txn + "\n" + build_distractor_log(
            avail_r, avail_d, n=1, start_mins=primary_mins + 5)

        calls = (
            [{"name": "toggle_lights", "args": {"room": r, "state": light_rev}}
             for r in log_lights] +
            [{"name": "lock_door", "args": {"door": d, "state": door_rev}}
             for d in log_doors]
        )
        resp = (
            " ".join(f"{ROOM_DISPLAY[r].title()} light {light_rev}."
                     for r in log_lights) + " " +
            " ".join(f"{DOOR_DISPLAY[d].title()} {door_rev_aw}."
                     for d in log_doors)
        )

        n_total     = len(calls)
        light_names = ", ".join(ROOM_DISPLAY[r] for r in log_lights)
        door_names  = ", ".join(DOOR_DISPLAY[d] for d in log_doors)
        t_label     = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"
        prompt      = random.choice(undo_phrases)

        # KEY CHANGE: enumerate sub-actions by tool type before emitting,
        # structurally identical to how gen_compound_three_action works.
        light_rev_calls = ", ".join(
            f"toggle_lights({r}, {light_rev})" for r in log_lights)
        door_rev_calls  = ", ".join(
            f"lock_door({d}, {door_rev})" for d in log_doors)

        think = (
            f"User said '{prompt}'. "
            f"'Undo'/'them'/'revert' → first [...] block ({t_label}). "
            f"First block contains MIXED device types . "
            f"Counting sub-actions: "
            f"({', '.join(str(i+1) for i in range(n_lights))}) "
            f"{n_lights} toggle_lights call(s): {light_rev_calls}. "
            f"({', '.join(str(n_lights+i+1) for i in range(n_doors))}) "
            f"{n_doors} lock_door call(s): {door_rev_calls}."
        )
        # build_ex() will inject:
        # "Total: N tool calls required. Emitting all N. ACTION REQUIRED."

        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think,
            category="mixed_device_undo"))

    return examples


def gen_scope_isolation_stress(target: int = 2_000) -> list:
    examples = []
    
    light_on_phrases = ["on all the lights", "turn on all lights", "on those lights"]
    light_off_phrases = [
        "off the lights that are on", "turn off all the lights",
        "kill the lights", "all lights off", "off those lights that are on",
        "off all those lights"
    ]
    
    door_lock_phrases = ["close all the doors", "lock all the doors", "close the doors that are open", "lock all the doors that are unlocked"]
    door_unlock_phrases = ["open all the doors", "unlock all doors"]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=4)
        state = generate_random_state(avail_r, avail_d)
        
        is_light_cmd = random.random() < 0.5
        
        if is_light_cmd:
            n_log_doors = random.randint(2, min(5, len(avail_d)))
            log_doors = random.sample(avail_d, n_log_doors)
            log_s = random.choice(["lock", "unlock"])
            log_aw = "locked" if log_s == "lock" else "unlocked"
            call_strs = [f"lock_door(door={d}, state={log_s})" for d in log_doors]
            summary = " ".join(f"{DOOR_DISPLAY[d].title()} {log_aw}." for d in log_doors)
            primary_txn = fmt_txn(random.randint(1,3), call_strs, summary)
            action_log = primary_txn
            
            cmd_s = random.choice(["on", "off"])
            opp_s = "off" if cmd_s == "on" else "on"
            action_lights = [r for r in avail_r if state["lights"][r]["state"] == opp_s]
            if not action_lights:
                state["lights"][avail_r[0]]["state"] = opp_s
                action_lights = [avail_r[0]]
            
            prompt = random.choice(light_on_phrases if cmd_s == "on" else light_off_phrases)
            
            calls = [{"name": "toggle_lights", "args": {"room": r, "state": cmd_s}} for r in action_lights]
            resp = " ".join(f"{ROOM_DISPLAY[r].title()} light {cmd_s}." for r in action_lights)
            
            l_summary = ", ".join(f"{r}:{state['lights'][r]['state']}" for r in avail_r)
            names_str = ", ".join(ROOM_DISPLAY[r] for r in action_lights)
            
            think = (
                f"User said '{prompt}'. "
                f"Single-device bulk command — lights only. "
                f"Checking ALL connected lights: {l_summary}. "
                f"{len(action_lights)} light(s) need to turn {cmd_s} ({names_str}). "
                f"Issuing {len(action_lights)} toggle_lights calls."
            )
            
        else:
            n_log_lights = random.randint(2, min(5, len(avail_r)))
            log_lights = random.sample(avail_r, n_log_lights)
            log_ls = random.choice(["on", "off"])
            call_strs = [f"toggle_lights(room={r}, state={log_ls})" for r in log_lights]
            summary = " ".join(f"{ROOM_DISPLAY[r].title()} light {log_ls}." for r in log_lights)
            primary_txn = fmt_txn(random.randint(1,3), call_strs, summary)
            action_log = primary_txn
            
            cmd_s = random.choice(["lock", "unlock"])
            aw = "locked" if cmd_s == "lock" else "unlocked"
            opp_aw = "unlocked" if cmd_s == "lock" else "locked"
            action_doors = [d for d in avail_d if state["doors"][d] == opp_aw]
            if not action_doors:
                state["doors"][avail_d[0]] = opp_aw
                action_doors = [avail_d[0]]
            
            prompt = random.choice(door_lock_phrases if cmd_s == "lock" else door_unlock_phrases)
            calls = [{"name": "lock_door", "args": {"door": d, "state": cmd_s}} for d in action_doors]
            resp = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in action_doors)

            d_summary = ", ".join(f"{DOOR_DISPLAY[d]}:{state['doors'][d]}" for d in avail_d)
            door_names_str = ", ".join(DOOR_DISPLAY[d] for d in action_doors)
            think = (
                f"User said '{prompt}'. "
                f"Single-device bulk command — doors only. "
                f"Checking ALL connected doors: {d_summary}. "
                f"{len(action_doors)} door(s) need to {cmd_s} ({door_names_str}). "
                f"Issuing {len(action_doors)} lock_door calls."
            )
        
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think, category="scope_isolation_stress"))
    
    return examples


def gen_response_text_grounding(target: int = 1_500) -> list:
    """
    Trains: response text must ONLY mention devices that were
    actually called as tool calls. No fabricated confirmations.
    
    Closes pattern 7 (turns 90, 10, 71): model generates
    "Living Room Door locked" in response when no door call was made.
    """
    examples = []
    
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=4)
        state = generate_random_state(avail_r, avail_d)
        
        # Mix: some lights need action, some doors already satisfied
        s = random.choice(["on", "off"])
        opp = "off" if s == "on" else "on"
        
        # Pick rooms that need action
        need_action = random.sample(avail_r, random.randint(2, min(4, len(avail_r))))
        already_done = random.sample(
            [r for r in avail_r if r not in need_action],
            random.randint(0, min(2, len(avail_r) - len(need_action)))
        )
        
        for r in need_action:
            apply_force(state, {"lights": {r: opp}}, avail_r, avail_d)
        for r in already_done:
            apply_force(state, {"lights": {r: s}}, avail_r, avail_d)
        
        calls = [{"name": "toggle_lights", "args": {"room": r, "state": s}}
                 for r in need_action]
        
        # Response ONLY mentions called rooms
        resp = " ".join(
            f"{ROOM_DISPLAY[r].title()} light {s}." for r in need_action)
        if already_done:
            resp += " " + " ".join(
                f"The {ROOM_DISPLAY[r]} light was already {s}."
                for r in already_done)
        
        # Add a distractor log with doors
        n_log_doors = random.randint(2, min(4, len(avail_d)))
        log_doors = random.sample(avail_d, n_log_doors)
        log_ds = random.choice(["lock", "unlock"])
        log_aw = "locked" if log_ds == "lock" else "unlocked"
        call_strs = [f"lock_door(door={d}, state={log_ds})" for d in log_doors]
        summary = " ".join(f"{DOOR_DISPLAY[d].title()} {log_aw}." for d in log_doors)
        action_log = fmt_txn(random.randint(1, 3), call_strs, summary)
        
        prompt = random.choice([
            "Off the lights that are on.", "On all the lights.", "Turn off all lights.",
            "Kill the lights.", "All lights off.", "Lights on please."
        ])

        l_summary = ", ".join(f"{r}:{state['lights'][r]['state']}" for r in avail_r)
        
        need_names = ", ".join(ROOM_DISPLAY[r] for r in need_action)
        done_names = (", ".join(ROOM_DISPLAY[r] for r in already_done)
                      if already_done else "none")
        log_door_names = ", ".join(DOOR_DISPLAY[d] for d in log_doors)
        
        think = (
            f"User said '{prompt}'. "
            f"Checking ALL connected lights: {l_summary}. "
            f"{len(calls)} light(s) currently {opp} ({need_names}). "
            + (f"{len(already_done)} light(s) already {s} ({done_names}). " if already_done else "")
            + f"Issuing {len(calls)} toggle_lights(state={s}) calls."
        )
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think,
            category="response_text_grounding"))
    
    return examples

def gen_direct_command_ignore_log(target: int = 1_500) -> list:
    """
    Trains: Direct commands (like 'next song', 'stop the music') should 
    trigger standard resolution, naturally ignoring the action log even if 
    it contains massive, recent light/door blocks.
    """
    examples = []
    
    direct_phrases = [
        ("stop the song", "stop"), ("stop the music", "stop"), ("pause the speaker", "pause"),
        ("next song", "next"), ("skip", "next"), ("next track", "next"),
        ("previous song", "previous"), ("go back a song", "previous")
    ]
    
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=4)
        state = generate_random_state(avail_r, avail_d)
        
        # Scenario: Speaker
        spk_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not spk_rooms: continue
        target_room = spk_rooms[0]
        
        prompt, action = random.choice(direct_phrases)
        
        # --- Build Dynamic, Multi-Block Distractor Logs ---
        log_blocks = []
        current_mins = random.randint(1, 3)
        
        if random.random() < 0.5:
            # Distractor: Lights
            n_lights = random.randint(3, len(avail_r))
            distract_rooms = random.sample(avail_r, n_lights)
            ls = random.choice(["on", "off"])
            call_strs = [f"toggle_lights(room={r}, state={ls})" for r in distract_rooms]
            summary = " ".join(f"{ROOM_DISPLAY[r].title()} light {ls}." for r in distract_rooms)
        else:
            # Distractor: Doors
            n_doors = random.randint(3, len(avail_d))
            distract_doors = random.sample(avail_d, n_doors)
            ds = random.choice(["lock", "unlock"])
            aw = "locked" if ds == "lock" else "unlocked"
            call_strs = [f"lock_door(door={d}, state={ds})" for d in distract_doors]
            summary = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in distract_doors)
            
        log_blocks.append(fmt_txn(current_mins, call_strs, summary))
        
        # Add an older speaker action dynamically
        current_mins += random.randint(1, 5)
        music = random.choice(LOCAL_MUSIC)
        spk_call = [f"control_speaker(room={target_room}, action=play, media=\"{music}\")"]

        spk_summary = f"Playing '{music}' on the {ROOM_DISPLAY[target_room]} speaker."
        
        log_blocks.append(fmt_txn(current_mins, spk_call, spk_summary))
        action_log = "\n".join(log_blocks)
        
        # --- Enforce Strict State for Rule 1 or 3 Resolution ---
        spk_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not spk_rooms: continue
        target_room = spk_rooms[0]
        
        state["speaker"][target_room] = "playing" 
        for r in spk_rooms[1:]:
            state["speaker"][r] = "stopped"  # target_room is ONLY eligible speaker
            
        calls = [{"name": "control_speaker", "args": {"room": target_room, "action": action}}]
        
        # Fix 1: Stop 'pauseped'
        if action in ["next", "previous"]:
            action_text = action
        elif action == "pause":
            action_text = "paused"
        elif action == "play":
            action_text = "playing"
        else:
            action_text = f"{action}ped"

        resp = f"{ROOM_DISPLAY[target_room].title()} speaker {action_text}."
        
        # --- Grounded, Topology-Aware Think Trace ---
        spk_list_str = ", ".join(spk_rooms)
        
        if len(spk_rooms) == 1:
            # Rule 1 applies
            think = (
                f"User said '{prompt}'. "
                f"Checking CONNECTED SPEAKERS: [{', '.join(spk_rooms)}]. "
                f"Exactly one connected ({target_room}). "
                f"Resolving to {target_room} automatically. "
                f"Calling control_speaker(room={target_room}, action={action})."
            )
        else:
            # Rule 3 applies
            think = (
                f"User said '{prompt}'. "
                f"Checking CONNECTED SPEAKERS: [{', '.join(spk_rooms)}]. "
                f"Multiple speakers connected. "
                f"STATE shows {target_room} is the only one in eligible state for '{action}'. "
                f"Inferring {target_room}. "
                f"Calling control_speaker(room={target_room}, action={action})."
            )
        
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            action_log=action_log, think_trace=think, category="direct_command_ignore_log"))
            
    return examples

# SFT ADDITION  END
def gen_local_media_commands(target: int = 2000) -> list:
    """
    Personalized media generation using the user's actual local music library.
    Handles both implicit resolution AND explicit room requests.
    Covers explicit room + Rules 1, 2, 3 of gadget resolution.

    Fixes applied:
    - room_clause pattern in Rule 3 (prevents "User is in ''" traces)
    - Full call signature including media param in every think trace
    - Consistent with all FIX-B positive-only trace requirements
    """
    examples = []

    # Implicit phrases (user doesn't specify room)
    play_phrases_implicit = [
        "Play {m}.", "Can you play {m}?", "Put on {m}.", "I want to hear {m}.","Put for me {m}.",
        "let's Listen to {m}.", "Queue up {m} for me.", "Play me {m}.", "Queue up {m} for me.",
        "play {m} for me.",
    ]

    # Explicit phrases (user names the room)
    play_phrases_explicit = [
        "Play {m} in the {r} for me.","Play {m} in the {r}.", "Put on {m} on the {r} speaker.",
        "Can you play {m} in the {r}?", "I want to hear {m} in the {r}.","Put for me {m} on the {r} speaker.",
        "Start playing {m} in the {r}.","I want you to play {m} in the {r} for me.",
    ]

    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3)
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not sp_rooms:
            avail_r.append("living_room")
            sp_rooms = ["living_room"]

        media = random.choice(LOCAL_MUSIC)
        rule = random.choices([0, 1, 2, 3], weights=[25, 25, 25, 25])[0]

        # FIX: Prevent Rule 2/3 from firing if only 1 speaker exists
        if rule in [2, 3] and len(sp_rooms) < 2:
            rule = 1

        # FIX: If Rule 1, strictly prune topology so 'Exactly one' trace is mathematically true
        if rule == 1:
            target_room = sp_rooms[0]
            avail_r = [r for r in avail_r if r not in SPEAKER_ROOMS or r == target_room]
            sp_rooms = [target_room]

        state = generate_random_state(avail_r, avail_d)
        base_think = f"Checking CONNECTED SPEAKERS from system prompt: [{', '.join(sp_rooms)}]. "
        user_room = ""
        target_room = ""
        prompt = ""
        think = ""

        if rule == 0:
            target_room = random.choice(sp_rooms)
            alias = random.choice(ROOM_ALIASES.get(target_room, [target_room.replace("_", " ")]))
            prompt = random.choice(play_phrases_explicit).format(m=media, r=alias)
            user_room = random.choice(avail_r + ["", ""])
            state["speaker"][target_room] = "stopped"
            think = base_think + (
                f"User said '{prompt}'. Requested specific media: '{media}'. "
                f"User explicitly named the '{alias}' speaker. "
                f"Resolving to {target_room}. "
                f"Calling control_speaker(room={target_room}, action=play, media=\"{media}\")."
            )

        elif rule == 1:
            target_room = sp_rooms[0]
            state["speaker"][target_room] = "stopped"
            prompt = random.choice(play_phrases_implicit).format(m=media)
            non_sp = [r for r in avail_r if r != target_room]
            user_room = random.choice(non_sp + ["", ""]) if non_sp else ""
            think = base_think + (
                f"User said '{prompt}'. Requested specific media: '{media}'. "
                f"Exactly one speaker connected ({target_room}). "
                f"Resolving to {target_room} automatically despite "
                f"current_user_room='{user_room}'. "
                f"Calling control_speaker(room={target_room}, action=play, "
                f"media=\"{media}\")."
            )

        elif rule == 2:
            target_room = random.choice(sp_rooms)
            user_room = target_room
            state["speaker"][target_room] = "stopped"
            prompt = random.choice(play_phrases_implicit).format(m=media)
            think = base_think + (
                f"User said '{prompt}'. Requested specific media: '{media}'. "
                f"Multiple speakers connected. "
                f"current_user_room='{user_room}' has a speaker. "
                f"Resolving to {user_room}. "
                f"Calling control_speaker(room={target_room}, action=play, media=\"{media}\")."
            )

        else:
            target_room = sp_rooms[0]
            non_sp_rooms = [r for r in avail_r if r not in sp_rooms]
            user_room = random.choice(non_sp_rooms) if non_sp_rooms else ""
            for r in sp_rooms:
                state["speaker"][r] = "playing"
            state["speaker"][target_room] = "stopped"
            prompt = random.choice(play_phrases_implicit).format(m=media)

            room_clause = (
                f"User is in '{user_room}' which has no speaker."
                if user_room else "User's location is unknown."
            )
            think = base_think + (
                f"User said '{prompt}'. Requested specific media: '{media}'. "
                f"Multiple speakers connected. {room_clause} "
                f"Checking states: exactly ONE speaker ({target_room}) is in the eligible "
                f"state ('stopped') for 'play'. "
                f"Inferring {target_room}. "
                f"Calling control_speaker(room={target_room}, action=play, media=\"{media}\")."
            )

        resp = f"Playing '{media}' on the {ROOM_DISPLAY[target_room]} speaker."
        calls = [{"name": "control_speaker", "args": {"room": target_room, "action": "play", "media": media}}]

        examples.append(build_ex(
            prompt, calls, resp, avail_r, avail_d, state,
            user_room=user_room, think_trace=think, category="local_media_commands"
        ))

    return examples


def gen_compound_media_plus_devices(target: int = 6500) -> list:
    examples = []

    play_phrases_implicit = [
        "Play {m}.", "Can you play {m}?", "put on {m}.", "I want to hear {m}.","Put for me {m}.",
        "let's Listen to {m}.", "Queue up {m} for me.", "play me {m}.", "Queue up {m} for me.",
        "play {m} for me.",
    ]
    play_phrases_explicit = [
        "play {m} on the {r} speaker.", "put on {m} on the {r} speaker.",
        "play {m} in the {r}.", "play {m} in the {r} for me.",
    ]
    
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=4, min_doors=3)
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not sp_rooms:
            avail_r.append("living_room")
            sp_rooms = ["living_room"]

        state = generate_random_state(avail_r, avail_d)
        media = random.choice(LOCAL_MUSIC)
        scenario = random.randint(1, 4)

        calls = []
        resp_fragments = []
        action_log = ""

        # FIX-F Crash Fix: Scenarios 1 and 3 require a valid room for implicit "the light".
        # Scenarios 2 and 4 use explicit/logged targets, so user_room can safely be empty.
        if scenario in [1, 3]:
            user_room = random.choice(avail_r)
        else:
            user_room = random.choice(avail_r + ["", ""])

        # Pre-calculate music target variables
        music_target = ""
        music_status = "unclear"
        music_think = ""

        sp_str = ", ".join(sp_rooms)
        if len(sp_rooms) == 1:
            music_target = sp_rooms[0]
            music_status = "resolved"
            music_think = (
                f"Checking CONNECTED SPEAKERS: [{sp_str}]. Exactly one speaker connected. "
                f"Resolving to {music_target}. Calling control_speaker(room={music_target}, action=play, media=\"{media}\")."
            )
        elif user_room in sp_rooms:
            music_target = user_room
            music_status = "resolved"
            music_think = (
                f"Checking CONNECTED SPEAKERS: [{sp_str}]. Multiple speakers connected. "
                f"current_user_room='{user_room}' has a speaker. "
                f"Resolving to {user_room}. Calling control_speaker(room={music_target}, action=play, media=\"{media}\")."
            )
        else:
            room_clause = (
                f"User is in '{user_room}' which has no speaker."
                if user_room else "User's location is unknown."
            )
            eligible_rooms = [r for r in sp_rooms if state["speaker"][r] == "stopped"]

            if len(eligible_rooms) == 1:
                music_target = eligible_rooms[0]
                music_status = "resolved"
                music_think = (
                    f"Checking CONNECTED SPEAKERS: [{sp_str}]. Multiple speakers connected. "
                    f"Exactly one speaker ({music_target}) is in eligible state ('stopped'). "
                    f"Inferring {music_target}. Calling control_speaker(room={music_target}, action=play, media=\"{media}\")."
                )
            else:
                music_status = "ambiguous"
                music_think = (
                    f"Checking CONNECTED SPEAKERS: [{sp_str}]. Multiple speakers connected.  "
                    f"Multiple speakers are eligible (or none are stopped) — cannot infer target. "
                    f"Calling intent_unclear(incomplete)."
                )

        if scenario == 1:
            state["lights"][user_room]["state"] = "on"
            j = random.choice(play_phrases_implicit).format(m=media)
            prompt = f"Off the light and {j}"

            calls.append({"name": "toggle_lights", "args": {"room": user_room, "state": "off"}})
            resp_fragments.append(f"turned off the {ROOM_DISPLAY[user_room]} light")
            part1_think = f"(1) 'off the light' — current_user_room='{user_room}' → toggle_lights(room={user_room}, state=off)."

            if music_status == "resolved":
                calls.append({"name": "control_speaker", "args": {"room": music_target, "action": "play", "media": media}})
                resp_fragments.append(f"playing '{media}' in the {ROOM_DISPLAY[music_target]}")
            else:
                calls.append({"name": "intent_unclear", "args": {"reason": "incomplete"}})
            part2_think = f"(2) '{j}' — {music_think}"

            think = f"Compound request. Counting sub-actions: {part1_think} {part2_think}"

        elif scenario == 2:
            d_room = random.choice(avail_d)
            state["doors"][d_room] = "unlocked"
            action_log = fmt_txn(2, [f"lock_door(door={d_room}, state=unlock)"], f"{d_room} door opened.")
            j = random.choice(play_phrases_implicit).format(m=media)
            prompt = f"Close that door and {j}"

            calls.append({"name": "lock_door", "args": {"door": d_room, "state": "lock"}})
            resp_fragments.append(f"locked the {d_room.replace('_', ' ')} door")
            part1_think = (
                f"(1) 'that door' — pronoun references first [...] block in RECENT ACTIONS "
                f"which shows {d_room} was just opened → lock_door(door={d_room}, state=lock)."
            )

            if music_status == "resolved":
                calls.append({"name": "control_speaker", "args": {"room": music_target, "action": "play", "media": media}})
                resp_fragments.append(f"started '{media}' on the {ROOM_DISPLAY[music_target]} speaker")
            else:
                calls.append({"name": "intent_unclear", "args": {"reason": "incomplete"}})
            part2_think = f"(2) '{j}' — {music_think}"

            think = f"Compound request. Counting sub-actions: {part1_think} {part2_think}"

        elif scenario == 3:
            d_room = random.choice(avail_d)
            state["lights"][user_room]["state"] = "on"
            state["doors"][d_room] = "unlocked"
            j = random.choice(play_phrases_implicit).format(m=media)
            prompt = f"Turn off the light, lock the {d_room.replace('_', ' ')} door, and {j}"

            calls.append({"name": "toggle_lights", "args": {"room": user_room, "state": "off"}})
            resp_fragments.append(f"the {ROOM_DISPLAY[user_room]} light is off")
            part1_think = f"(1) 'off the light' — current_user_room='{user_room}' → toggle_lights(room={user_room}, state=off)."

            calls.append({"name": "lock_door", "args": {"door": d_room, "state": "lock"}})
            resp_fragments.append(f"the {d_room.replace('_', ' ')} door is locked")
            part2_think = f"(2) '{d_room.replace('_', ' ')} door' is explicitly named → lock_door(door={d_room}, state=lock)."

            if music_status == "resolved":
                calls.append({"name": "control_speaker", "args": {"room": music_target, "action": "play", "media": media}})
                resp_fragments.append(f"playing '{media}'")
            else:
                calls.append({"name": "intent_unclear", "args": {"reason": "incomplete"}})
            part3_think = f"(3) '{j}' — {music_think}"

            think = f"Compound request. Counting sub-actions: {part1_think} {part2_think} {part3_think}"

        else:
            l_room = random.choice(avail_r)
            s_room = music_target if music_status == "resolved" else random.choice(sp_rooms)
            state["lights"][l_room]["state"] = "on"
            l_alias = random.choice(ROOM_ALIASES[l_room])
            s_alias = random.choice(ROOM_ALIASES[s_room])
            
            j = random.choice(play_phrases_explicit).format(m=media, r=s_alias)
            prompt = f"Turn off the {l_alias} light and {j}"

            calls.append({"name": "toggle_lights", "args": {"room": l_room, "state": "off"}})
            resp_fragments.append(f"turned off the {l_alias.title()} light")
            part1_think = f"(1) explicitly named '{l_alias} light' → toggle_lights(room={l_room}, state=off)."

            calls.append({"name": "control_speaker", "args": {"room": s_room, "action": "play", "media": media.lower()}})
            resp_fragments.append(f"playing '{media}' on the {s_alias.title()} speaker")
            part2_think = f"(2) explicitly named '{s_alias} speaker' → control_speaker(room={s_room}, action=play, media=\"{media.lower()}\")."
            
            music_status = "resolved" 
            think = f"Compound request. Counting sub-actions: {part1_think} {part2_think}"

        if len(resp_fragments) > 1:
            main_resp = ", ".join(resp_fragments[:-1]) + f", and {resp_fragments[-1]}."
        elif resp_fragments:
            main_resp = f"I've {resp_fragments[0]}."
        else:
            main_resp = ""

        main_resp = main_resp[0].upper() + main_resp[1:] if main_resp else ""
        if music_status == "ambiguous":
            response = f"{main_resp} However, I'm not sure which speaker to use for the music. Which one did you mean?"
        else:
            response = main_resp

        examples.append(build_ex(
            prompt, calls, response, avail_r, avail_d, state,
            user_room=user_room, action_log=action_log,
            think_trace=think, category="compound_media_plus_devices"
        ))

    return examples

def gen_mixed_compound_hallway_boost(target: int = 600) -> list:
    """
    Targeted fix for hallway/office missing-call failures in mixed_compound.
    Forces these rooms into compound commands so the model sees them
    explicitly in this context at sufficient frequency.
    """
    examples = []
    UNDERREPRESENTED = ["hallway", "office", "bathroom"]
    
    for _ in range(target):
        forced_room = random.choice(UNDERREPRESENTED)
        other_room  = random.choice([r for r in ALL_ROOMS if r != forced_room])
        avail_r, avail_d = sample_topology(
            required_rooms=[forced_room, other_room], min_rooms=3)
        state = generate_random_state(avail_r, avail_d)
        
        ls1 = random.choice(["on", "off"])
        ls2 = random.choice(["on", "off"])
        d   = random.choice(avail_d)
        ds  = random.choice(["lock", "unlock"])
        aw  = "locked" if ds == "lock" else "unlocked"
        
        apply_force(state, {
            "lights": {
                forced_room: "off" if ls1 == "on" else "on",
                other_room:  "off" if ls2 == "on" else "on",
            },
            "doors": {d: "unlocked" if ds == "lock" else "locked"}
        }, avail_r, avail_d)
        
        fa = random.choice(ROOM_ALIASES[forced_room])
        oa = random.choice(ROOM_ALIASES[other_room])
        da = random.choice(DOOR_ALIASES[d])
        
        calls = [
            {"name": "toggle_lights", "args": {"room": forced_room, "state": ls1}},
            {"name": "toggle_lights", "args": {"room": other_room,  "state": ls2}},
            {"name": "lock_door",     "args": {"door": d, "state": ds}},
        ]
        
        # VARY USER ROOM: 
        # 25% Empty, 25% in the Hallway, 50% in some completely different room
        user_room = random.choice(["", forced_room, other_room, random.choice(avail_r)])
        
        think = (
            f"Compound request. Counting sub-actions: "
            f"(1) toggle_lights(room={forced_room}, state={ls1}) . "
            f"(2) toggle_lights(room={other_room}, state={ls2}), "
            f"(3) lock_door(door={d}, state={ds}). "
            f"Total: 3 tool calls required. Emitting all 3."
        )
        prompt = (f"Turn {ls1} the {fa} light, {ls2} the {oa} light, "
                  f"and {ds} the {da}.")
        resp = f"{fa.title()} light {ls1}. {oa.title()} light {ls2}. {da.title()} {aw}."
        
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=user_room,  # <--- PASS IT HERE
            think_trace=think, category="mixed_compound_hallway_boost"))
    
    return examples


    
def gen_action_required_no_ambiguity(target: int = 2_000) -> list:
    """
    Explicit room + state mismatch = toggle_lights. Always.
    The model must never call intent_unclear for a fully-specified request.
    """
    examples = []
    on_t  = [
        "Turn on the {r} light.", "Switch on the {r} light.",
        "On the {r} light.", "The {r} light on.",
        "I need the {r} light on.", "Can you turn on the {r} light?",
        "Light on in the {r}.", "{r} lights on.",
    ]
    off_t = [
        "Turn off the {r} light.", "Switch off the {r} light.",
        "Off the {r} light.", "The {r} light off.",
        "I need the {r} light off.", "Can you turn off the {r} light?",
        "Kill the {r} light.", "{r} lights off.",
    ]
 
    for _ in range(target):
        r = random.choice(ALL_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[r])
        state = generate_random_state(avail_r, avail_d)
        desired  = random.choice(["on", "off"])
        opposite = "off" if desired == "on" else "on"
        apply_force(state, {"lights": {r: opposite}}, avail_r, avail_d)
        alias     = random.choice(ROOM_ALIASES[r])
        prompt    = random.choice(on_t if desired == "on" else off_t).format(r=alias)
        user_room = random.choice(["", r, random.choice(avail_r)])
        # FIX-1: STATE check first, no defensive meta-commentary
        think = (
            f"Checking STATE: {r}={opposite}. "
            f"User explicitly named '{alias}' (room='{r}'). "
            f"User wants {desired}. "
            f"Mismatch — STATE shows {opposite}, user wants {desired}. "
            f"Calling toggle_lights(room={r}, state={desired})."
        )
        examples.append(build_ex(
            prompt,
            [{"name": "toggle_lights", "args": {"room": r, "state": desired}}],
            f"The {alias} light is now {desired}.",
            avail_r, avail_d, state,
            user_room=user_room,
            think_trace=think, category="action_required_no_ambiguity"))
 
    return examples

def gen_multi_room_doors(target: int = 1_500) -> list:
    examples = []
    lock_t   = ["Lock the {d_list} doors.", "Close the {d_list} doors.", "Secure the {d_list} doors."]
    unlock_t = ["Unlock the {d_list} doors.", "Open the {d_list} doors.", "Open up the {d_list} doors."]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_doors=5)
        n = random.choices([2, 3, 4], weights=[40, 40, 20])[0]
        n = min(n, len(avail_d))
        if n < 2: continue
        chosen = random.sample(avail_d, n)
        state  = generate_random_state(avail_r, avail_d)
        s   = random.choice(["lock", "unlock"])
        opp = "unlock" if s == "lock" else "lock"
        aw  = "locked" if s == "lock" else "unlocked"
        for d in chosen:
            apply_force(state, {"doors": {d: opp}}, avail_r, avail_d)
        aliases = [random.choice(DOOR_ALIASES[d]).replace(" door", "") for d in chosen]
        d_list  = (f"{aliases[0]} and {aliases[1]}" if n == 2 else ", ".join(aliases[:-1]) + f", and {aliases[-1]}")
        prompt  = random.choice(lock_t if s == "lock" else unlock_t).format(d_list=d_list)
        calls   = [{"name": "lock_door", "args": {"door": d, "state": s}} for d in chosen]
        resp    = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in chosen)
        traces  = ", ".join(f"lock_door({d}, {s})" for d in chosen)
        think   = (
            f"User listed {n} doors: {', '.join(chosen)}. "
            f"'{'Open' if s=='unlock' else 'Close/lock'}' maps to state='{s}'. "
            f"Issuing exactly {n} lock_door calls (one per door): {traces}."
        )
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            think_trace=think, category="multi_room_doors"))
    return examples


def gen_compound_three_action_local(target: int = 1_000) -> list:
    examples = []
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]
    d_verbs = ["close", "lock", "shut", "secure", "open", "unlock"]
    l_verbs = ["on", "off", "turn on", "turn off"]
    m_tmpl  = ["play {m} for me", "put on {m}", "play {m}", "start {m}"]
    for _ in range(target):
        u_room = random.choice(ROOM_DOOR_ROOMS)
        avail_r, avail_d = sample_topology(
            required_rooms=[u_room], required_doors=[u_room], min_rooms=3)
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not sp_rooms: continue
        state = generate_random_state(avail_r, avail_d)

        ls = random.choice(["on", "off"])
        apply_force(state, {"lights": {u_room: "off" if ls == "on" else "on"}}, avail_r, avail_d)
        l_call = {"name": "toggle_lights", "args": {"room": u_room, "state": ls}}
        l_ph   = random.choice([f"{ls} the light", f"turn {ls} the light", f"{ls} this light"])

        ds = random.choice(["lock", "unlock"])
        apply_force(state, {"doors": {u_room: "unlocked" if ds == "lock" else "locked"}}, avail_r, avail_d)
        d_aw   = "locked" if ds == "lock" else "unlocked"
        d_call = {"name": "lock_door", "args": {"door": u_room, "state": ds}}
        d_ph   = random.choice([f"{'close' if ds=='lock' else 'open'} the door", f"{ds} this door"])

        media = random.choice(LOCAL_MUSIC)
        if len(sp_rooms) == 1:
            sp_t = sp_rooms[0]
            m_think = f"Checking CONNECTED SPEAKERS: [{sp_t}]. Exactly one connected. Rule 1 → resolving to {sp_t}."
        elif u_room in sp_rooms:
            sp_t = u_room
            m_think = f"Multiple speakers. current_user_room='{u_room}' has speaker. Rule 2 → resolving to {u_room}."
        else:
            sp_t = sp_rooms[0]
            for r in sp_rooms: state["speaker"][r] = "playing"
            state["speaker"][sp_t] = "stopped"
            m_think = f"Multiple speakers. Only '{sp_t}' eligible (stopped). Rule 3 → inferring {sp_t}."
        
        state["speaker"][sp_t] = "stopped"
        m_call = {"name": "control_speaker", "args": {"room": sp_t, "action": "play", "media": media}}
        m_ph   = random.choice(m_tmpl).format(m=media)

        calls  = [d_call, l_call, m_call]
        prompt = random.choice([f"{d_ph.capitalize()}, {l_ph}, and {m_ph}.", f"{l_ph.capitalize()}, {d_ph}, and {m_ph}."])
        resp   = (f"{DOOR_DISPLAY[u_room].title()} {d_aw}. {ROOM_DISPLAY[u_room].title()} light {ls}. "
                  f"Playing '{media}' on the {ROOM_DISPLAY[sp_t]} speaker.")
        think  = (
            f"Compound request. Counting sub-actions: "
            f"(1) '{d_ph}' — current_user_room='{u_room}' → lock_door(door={u_room}, state={ds}). "
            f"(2) '{l_ph}' — current_user_room='{u_room}' → toggle_lights(room={u_room}, state={ls}). "
            f"(3) '{m_ph}' — {m_think} Calling control_speaker(room={sp_t}, action=play, media='{media}')."
        )
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=u_room, think_trace=think, category="compound_three_action_local"))
    return examples


def gen_open_unlock_direction_reinforced(target: int = 1_500) -> list:
    examples = []
    bulk_p = ["Open all the doors.", "Unlock all doors.", "Open every door.", "Unlock everything."]
    single_p = ["Open this door.", "Unlock this door.", "Open the {d}.", "Unlock the {d}."]
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_doors=4)
        state = generate_random_state(avail_r, avail_d)
        is_bulk = random.random() < 0.55

        if is_bulk:
            for d in avail_d: apply_force(state, {"doors": {d: "locked"}}, avail_r, avail_d)
            prompt = random.choice(bulk_p)
            calls  = [{"name": "lock_door", "args": {"door": d, "state": "unlock"}} for d in avail_d]
            resp   = " ".join(f"{DOOR_DISPLAY[d].title()} unlocked." for d in avail_d)
            ss     = ", ".join(f"{DOOR_DISPLAY[d]}:locked" for d in avail_d)
            think  = (
                f"User said '{prompt}'. "
                f"'Open'/'unlock' maps to target state='unlock'. "
                f"Checking ALL connected doors: {ss}. "
                f"All {len(avail_d)} are currently locked. "
                f"Issuing {len(avail_d)} lock_door(state=unlock) calls."
            )
        else:
            d = random.choice(avail_d)
            apply_force(state, {"doors": {d: "locked"}}, avail_r, avail_d)
            alias = random.choice(DOOR_ALIASES[d])
            valid_ur = [r for r in ROOM_DOOR_ROOMS if r in avail_r] + [""]
            u_room = (d if d in ROOM_DOOR_ROOMS and d in avail_r else random.choice(valid_ur))
            prompt = ("Open this door." if u_room == d else random.choice(single_p).format(d=alias))
            calls  = [{"name": "lock_door", "args": {"door": d, "state": "unlock"}}]
            resp   = f"{alias.title()} unlocked."
            think  = (
                f"User said '{prompt}'. Target door: '{d}'. "
                f"STATE shows {d}=locked. "
                f"'Open'/'unlock' maps to lock_door(state=unlock). "
                f"Calling lock_door(door={d}, state=unlock)."
            )
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room="" if is_bulk else u_room, think_trace=think, category="open_unlock_reinforced"))
    return examples

def gen_compound_thermostat_incremental(target: int = 2_000) -> list:
    examples = []
    inc_p = ["increase temp by {n}", "raise the temp by {n}", "bump it up {n} degrees"]
    dec_p = ["decrease temp by {n}", "lower the temp by {n}", "drop it {n} degrees"]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3)
        state = generate_random_state(avail_r, avail_d)
        cur   = state["thermostat"]["temperature"]
        cur_mode = state["thermostat"]["mode"]
        direc = random.choice(["up", "down"])
        n     = random.randint(1, 20)
        new_val = max(MIN_T, min(MAX_T, (cur + n) if direc == "up" else (cur - n)))
        mode    = "heat" if new_val > cur else "cool" if new_val < cur else cur_mode
        t_ph    = random.choice(inc_p if direc == "up" else dec_p).format(n=n)
        t_call  = {"name": "set_thermostat", "args": {"temperature": new_val, "mode": mode}}
        t_think = f"'{t_ph}' — relative increment: {cur} {'+ ' if direc == 'up' else '- '}{n} = {new_val}F (clamped). set_thermostat(temperature={new_val}, mode='{mode}')."

        combo = random.choice(["light", "door"])
        if combo == "light":
            r  = random.choice(avail_r)
            ls = random.choice(["on", "off"])
            apply_force(state, {"lights": {r: "off" if ls=="on" else "on"}}, avail_r, avail_d)
            alias  = random.choice(ROOM_ALIASES[r])
            l_ph   = f"turn {ls} the {alias} light"
            l_call = {"name": "toggle_lights", "args": {"room": r, "state": ls}}
            l_think = f"'{l_ph}' — explicit room '{r}' named — toggle_lights(room={r}, state={ls})."
            calls  = [l_call, t_call]
            prompt = random.choice([f"{l_ph.capitalize()} and {t_ph}.", f"{t_ph.capitalize()} and {l_ph}."])
            resp = f"The {alias} light is now {ls}. Thermostat set to {new_val}°F in {mode} mode."
            think = f"Compound request. Counting sub-actions: (1) {l_think} (2) {t_think}"
        else:
            d  = random.choice(avail_d)
            ds = random.choice(["lock", "unlock"])
            apply_force(state, {"doors": {d: "unlocked" if ds=="lock" else "locked"}}, avail_r, avail_d)
            aw     = "locked" if ds == "lock" else "unlocked"
            da     = random.choice(DOOR_ALIASES[d])
            d_ph   = f"{'close' if ds=='lock' else 'open'} the {da}"
            d_call = {"name": "lock_door", "args": {"door": d, "state": ds}}
            d_think = f"'{d_ph}' — explicit door '{d}' named — lock_door(door={d}, state={ds})."
            calls  = [d_call, t_call]
            prompt = random.choice([f"{d_ph.capitalize()} and {t_ph}.", f"{t_ph.capitalize()} and {d_ph}."])
            resp = f"The {da} is now {aw}. Thermostat set to {new_val}°F in {mode} mode."
            think = f"Compound request. Counting sub-actions: (1) {d_think} (2) {t_think}"

        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=random.choice(["", random.choice(avail_r)]),
            think_trace=think, category="compound_thermostat_incremental"))
    return examples


def gen_compound_log_undo_plus_action(target: int = 1_500) -> list:
    examples = []
    undo_p = ["undo that and", "revert that and", "take that back and"]
    for _ in range(target):
        avail_r, avail_d = sample_topology(min_rooms=3, min_doors=3)
        state = generate_random_state(avail_r, avail_d)
        block_type = random.choice(["lights", "doors", "mixed"])
        call_strs, undo_calls, undo_resp = [], [], []
        primary_mins = random.randint(1, 4)
        t_label = f"{primary_mins} min{'s' if primary_mins > 1 else ''} ago"

        if block_type == "lights":
            n = random.randint(3, min(6, len(avail_r)))
            rooms = random.sample(avail_r, n)
            ls    = random.choice(["on", "off"])
            for r in rooms:
                apply_force(state, {"lights": {r: ls}}, avail_r, avail_d)
                call_strs.append(f"toggle_lights(room={r}, state={ls})")
                opp = "off" if ls == "on" else "on"
                undo_calls.append({"name": "toggle_lights", "args": {"room": r, "state": opp}})
                undo_resp.append(f"{ROOM_DISPLAY[r].title()} light {opp}.")
            summary = " ".join(f"{ROOM_DISPLAY[r].title()} light {ls}." for r in rooms)
            undo_think = f"First block ({t_label}): {n} lights ({', '.join(rooms)}) turned {ls}. Reversing all {n} to {opp}."

        elif block_type == "doors":
            n = random.randint(3, min(6, len(avail_d)))
            doors = random.sample(avail_d, n)
            ds    = random.choice(["lock", "unlock"])
            aw    = "locked" if ds == "lock" else "unlocked"
            for d in doors:
                apply_force(state, {"doors": {d: aw}}, avail_r, avail_d)
                call_strs.append(f"lock_door(door={d}, state={ds})")
                opp_ds = "unlock" if ds == "lock" else "lock"
                opp_aw = "unlocked" if ds == "lock" else "locked"
                undo_calls.append({"name": "lock_door", "args": {"door": d, "state": opp_ds}})
                undo_resp.append(f"{DOOR_DISPLAY[d].title()} {opp_aw}.")
            summary = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in doors)
            opp_ds  = "unlock" if ds == "lock" else "lock"
            undo_think = f"First block ({t_label}): {n} doors ({', '.join(doors)}) {ds}ed. Reversing all {n} to {opp_ds}."

        else:
            n_lights = random.randint(2, min(3, len(avail_r)))
            n_doors  = random.randint(1, min(2, len(avail_d)))
            rooms = random.sample(avail_r, n_lights)
            doors = random.sample(avail_d, n_doors)
            ls = random.choice(["on", "off"]); opp_l = "off" if ls == "on" else "on"
            ds = random.choice(["lock", "unlock"]); opp_ds = "unlock" if ds=="lock" else "lock"
            aw_l = ls; aw_d = "locked" if ds=="lock" else "unlocked"
            opp_aw_d = "unlocked" if ds=="lock" else "locked"
            for r in rooms:
                apply_force(state, {"lights": {r: ls}}, avail_r, avail_d)
                call_strs.append(f"toggle_lights(room={r}, state={ls})")
                undo_calls.append({"name": "toggle_lights", "args": {"room": r, "state": opp_l}})
                undo_resp.append(f"{ROOM_DISPLAY[r].title()} light {opp_l}.")
            for d in doors:
                apply_force(state, {"doors": {d: aw_d}}, avail_r, avail_d)
                call_strs.append(f"lock_door(door={d}, state={ds})")
                undo_calls.append({"name": "lock_door", "args": {"door": d, "state": opp_ds}})
                undo_resp.append(f"{DOOR_DISPLAY[d].title()} {opp_aw_d}.")
            summary = " ".join(f"{ROOM_DISPLAY[r].title()} light {ls}." for r in rooms) + " " + " ".join(f"{DOOR_DISPLAY[d].title()} {aw_d}." for d in doors)
            undo_think = f"First block ({t_label}): {n_lights} lights turned {ls}, {n_doors} door(s) {ds}ed. Reversing ALL items in the block."

        primary_txn = fmt_txn(primary_mins, call_strs, summary)
        action_log  = primary_txn + "\n" + build_distractor_log(avail_r, avail_d, n=1, start_mins=primary_mins + 5)

        second_type = random.choice(["thermostat", "light", "scene"])
        second_calls, second_resp, second_think = [], "", ""
        if second_type == "thermostat":
            cur = state["thermostat"]["temperature"]
            val = random.randint(MIN_T, MAX_T)
            while val == cur: val = random.randint(MIN_T, MAX_T)
            md  = "cool" if val < cur else "heat"
            second_calls = [{"name": "set_thermostat", "args": {"temperature": val, "mode": md}}]
            second_resp  = f"Thermostat set to {val}°F in {md} mode."
            second_think = f"Independent action: set_thermostat(temperature={val}, mode={md})."
            s_ph = f"set the temp to {val}"
        elif second_type == "light":
            r2 = random.choice(avail_r); ls2 = random.choice(["on", "off"])
            apply_force(state, {"lights": {r2: "off" if ls2=="on" else "on"}}, avail_r, avail_d)
            al2 = random.choice(ROOM_ALIASES[r2])
            second_calls = [{"name": "toggle_lights", "args": {"room": r2, "state": ls2}}]
            second_resp  = f"The {al2} light is now {ls2}."
            second_think = f"Independent action: toggle_lights(room={r2}, state={ls2})."
            s_ph = f"turn {ls2} the {al2} light"
        else:
            sc = random.choice(SCENES)
            state["active_scene"] = None
            second_calls = [{"name": "set_scene", "args": {"scene": sc}}]
            second_resp  = SCENE_RESP[sc]
            second_think = f"Independent action: set_scene(scene={sc})."
            s_ph = f"activate {sc.replace('_', ' ')}"

        all_calls = undo_calls + second_calls
        prompt = f"{random.choice(undo_p).capitalize()} {s_ph}."
        resp   = " ".join(undo_resp) + " " + second_resp
        n_total = len(all_calls)
        think  = f"Compound request. Part 1 (undo): {undo_think} Issuing {len(undo_calls)} reversal call(s). Part 2 (independent): {second_think} Total: {n_total} tool call(s)."
        
        examples.append(build_ex(prompt, all_calls, resp.strip(), avail_r, avail_d, state,
            user_room=random.choice(["", random.choice(avail_r)]), action_log=action_log, think_trace=think, category="compound_log_undo_plus_action"))
    return examples


def gen_compound_local_door_plus_media(target: int = 1_500) -> list:
    examples = []
    ROOM_DOOR_ROOMS = ["bedroom", "bathroom", "office", "kitchen", "living_room"]
    d_opts = ["close this door", "lock this door", "close the door", "lock the door"]
    m_opts = ["play {m}", "put on {m}", "listen to {m}"]
    for _ in range(target):
        u_room = random.choice(ROOM_DOOR_ROOMS)
        avail_r, avail_d = sample_topology(required_rooms=[u_room], required_doors=[u_room], min_rooms=3)
        sp_rooms = [r for r in avail_r if r in SPEAKER_ROOMS]
        if not sp_rooms: continue
        state = generate_random_state(avail_r, avail_d)
        apply_force(state, {"doors": {u_room: "unlocked"}}, avail_r, avail_d)
        
        media = random.choice(LOCAL_MUSIC)
        d_ph  = random.choice(d_opts)
        m_ph  = random.choice(m_opts).format(m=media)
        d_call = {"name": "lock_door", "args": {"door": u_room, "state": "lock"}}
        d_think = f"'{d_ph}' — current_user_room='{u_room}' → lock_door(door={u_room}, state=lock)."

        sp_str = ", ".join(sp_rooms)
        if len(sp_rooms) == 1:
            sp_t = sp_rooms[0]
            state["speaker"][sp_t] = "stopped"
            m_think = f"Checking CONNECTED SPEAKERS: [{sp_str}]. Exactly one speaker connected. Resolving to {sp_t}. control_speaker(room={sp_t}, action=play, media=\"{media}\")."
        elif u_room in sp_rooms:
            sp_t = u_room
            state["speaker"][sp_t] = "stopped"
            m_think = f"Checking CONNECTED SPEAKERS: [{sp_str}]. Multiple speakers. current_user_room='{u_room}' has speaker. Rule 2 → {sp_t}. control_speaker(room={sp_t}, action=play, media=\"{media}\")."
        else:
            sp_t = sp_rooms[0]
            for r in sp_rooms: state["speaker"][r] = "playing"
            state["speaker"][sp_t] = "stopped"
            m_think = f"Checking CONNECTED SPEAKERS: [{sp_str}]. Multiple speakers. Only '{sp_t}' in eligible state (stopped). Rule 3 → {sp_t}. control_speaker(room={sp_t}, action=play, media=\"{media}\")."
            
        m_call = {"name": "control_speaker", "args": {"room": sp_t, "action": "play", "media": media}}
        calls  = [d_call, m_call]
        prompt = random.choice([f"{d_ph.capitalize()} and {m_ph}.", f"{m_ph.capitalize()} and {d_ph}."])
        resp   = f"{DOOR_DISPLAY[u_room].title()} locked. Playing '{media}' on the {ROOM_DISPLAY[sp_t]} speaker."
        think  = f"Compound request. Counting sub-actions: (1) {d_think} (2) {m_think}"
        
        examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
            user_room=u_room, think_trace=think, category="compound_local_door_plus_media"))
    return examples


def gen_high_count_undo_stress(target: int = 1_000) -> list:
    """Fixes count truncation by forcing 7-9 devices in undo blocks."""
    examples = []
    undo_p = ["Undo that.", "Revert that.", "Reverse that.", "Take that back.", "Undo all that."]
    open_p = ["Open them.", "Unlock them.", "On them back.", "Reverse those."]
    for _ in range(target):
        device_type = random.choice(["lights", "doors"])
        if device_type == "lights":
            avail_r = list(ALL_ROOMS)
            avail_d = random.sample(ALL_DOORS, random.randint(3, 5))
            state = generate_random_state(avail_r, avail_d)
            ls    = random.choice(["on", "off"])
            opp   = "off" if ls == "on" else "on"
            for r in avail_r:
                apply_force(state, {"lights": {r: ls}}, avail_r, avail_d)
            call_strs = [f"toggle_lights(room={r}, state={ls})" for r in avail_r]
            summary   = " ".join(f"{ROOM_DISPLAY[r].title()} light {ls}." for r in avail_r)
            pm = random.randint(1, 4)
            primary_txn = fmt_txn(pm, call_strs, summary)
            action_log  = primary_txn + "\n" + build_distractor_log(
                avail_r, avail_d, n=2, start_mins=pm + 3)
            calls  = [{"name": "toggle_lights", "args": {"room": r, "state": opp}} for r in avail_r]
            resp   = " ".join(f"{ROOM_DISPLAY[r].title()} light {opp}." for r in avail_r)
            t_label = f"{pm} min{'s' if pm>1 else ''} ago"
            n_total = len(avail_r)
            think = (
                f"User said 'undo'. "
                f"First block ({t_label}): {n_total} lights ({', '.join(avail_r)}) turned {ls}. "
                f"Reversing ALL {n_total}. "
                f"Issuing exactly {n_total} toggle_lights(state={opp}) calls."
            )
            examples.append(build_ex(random.choice(undo_p), calls, resp, avail_r, avail_d, state,
                action_log=action_log, think_trace=think, category="high_count_undo_stress"))

        else:
            avail_r = random.sample(ALL_ROOMS, random.randint(3, 5))
            avail_d = list(ALL_DOORS)
            state   = generate_random_state(avail_r, avail_d)
            ds      = random.choice(["lock", "unlock"])
            aw      = "locked" if ds == "lock" else "unlocked"
            rev_s   = "unlock" if ds == "lock" else "lock"
            rev_aw  = "unlocked" if ds == "lock" else "locked"
            for d in avail_d:
                apply_force(state, {"doors": {d: aw}}, avail_r, avail_d)
            call_strs = [f"lock_door(door={d}, state={ds})" for d in avail_d]
            summary   = " ".join(f"{DOOR_DISPLAY[d].title()} {aw}." for d in avail_d)
            pm = random.randint(1, 4)
            primary_txn = fmt_txn(pm, call_strs, summary)
            action_log  = primary_txn + "\n" + build_distractor_log(
                avail_r, avail_d, n=2, start_mins=pm + 3)
            calls  = [{"name": "lock_door", "args": {"door": d, "state": rev_s}} for d in avail_d]
            resp   = " ".join(f"{DOOR_DISPLAY[d].title()} {rev_aw}." for d in avail_d)
            t_label = f"{pm} min{'s' if pm>1 else ''} ago"
            n_total = len(avail_d)
            prompt  = random.choice(open_p if rev_s == "unlock" else undo_p)
            think = (
                f"User said '{prompt}'. "
                f"First block ({t_label}): {n_total} doors ({', '.join(avail_d)}) {ds}ed. "
                f"Reversing ALL {n_total}. "
                f"Issuing exactly {n_total} lock_door(state={rev_s}) calls."
            )
            examples.append(build_ex(prompt, calls, resp, avail_r, avail_d, state,
                action_log=action_log, think_trace=think, category="high_count_undo_stress"))
    return examples
     
    
def gen_speaker_on_stop_synonyms(target: int = 3_000) -> list:
    examples = []
    ON_TO_PLAY = [
        "On the speaker.", "On the music.", "Turn on the music.",
        "Turn on the speaker.", "Switch on the speaker.", "Music on.",
        "On the {r} speaker.", "Turn on the {r} speaker.",
    ]
    OFF_TO_STOP = [
        "Off the speaker.", "Off the music.", "Turn off the music.",
        "Turn off the speaker.", "Kill the music.", "Music off.",
        "Off the {r} speaker.", "Turn off the {r} speaker.",
    ]
    for _ in range(target):
        r = random.choice([x for x in ALL_ROOMS if x in SPEAKER_ROOMS])
        avail_r, avail_d = sample_topology(required_rooms=[r])
        # Force single speaker so Rule 1 always applies and think trace is truthful
        avail_r = [x for x in avail_r if x not in SPEAKER_ROOMS or x == r]
        state = generate_random_state(avail_r, avail_d)
        is_play = random.random() < 0.55
        action = "play" if is_play else "stop"
        state["speaker"][r] = "stopped" if is_play else "playing"
        alias = random.choice(ROOM_ALIASES[r])
        tmpl  = random.choice(ON_TO_PLAY if is_play else OFF_TO_STOP)
        prompt = tmpl.format(r=alias) if "{r}" in tmpl else tmpl
        # VARY USER ROOM: The user can be anywhere, Rule 1 (only 1 speaker) still applies!
        user_room = random.choice(["", r, random.choice(avail_r)])

        spk_str = ", ".join(avail_r) # since we pruned avail_r to only have 1 speaker
        spk_list = [x for x in avail_r if x in SPEAKER_ROOMS]
        conn_str = ", ".join(spk_list)

        think = (
            f"User said '{prompt}'. "
            f"'On'/'turn on'/'switch on' for a speaker maps to action='play'. "
            f"'Off'/'turn off'/'kill' for a speaker maps to action='stop'. "
            f"Checking CONNECTED SPEAKERS: [{conn_str}]. Exactly one speaker connected. "
            f"Resolving automatically to {r}. "
            f"Calling control_speaker(room={r}, action={action})."
        )
        resp = (f"Playing music on the {alias} speaker." if action == "play"
                else f"Stopped the music on the {alias} speaker.")

        examples.append(build_ex(prompt,
            [{"name": "control_speaker", "args": {"room": r, "action": action}}],
            resp, avail_r, avail_d, state,
            user_room=user_room,
            think_trace=think, category="speaker_on_stop_synonyms"))
    return examples  # ← THE FIX: was missing entirely


# ══════════════════════════════════════════════════════════════════════
# MASTER ASSEMBLY
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# MASTER ASSEMBLY
# ══════════════════════════════════════════════════════════════════════

FINAL_TARGET = 160_000 

CATEGORY_PLAN = [
    # ── Core Generators ────────────────────────────────────────────────
    ("already_satisfied",        gen_already_satisfied,          1_800),
    ("action_required",          gen_action_required,            3_000),
    ("user_room_lights",         gen_user_room_lights,           3_000),
    ("user_room_doors",          gen_user_room_doors,            1_200),
    ("scenes",                   gen_scenes,                     3_000),
    ("thermostat",               gen_thermostat,                 1_800),
    ("status_queries",           gen_status_queries,             1_500),
    ("rejections",               gen_rejections,                 1_500),
    ("incomplete_no_room",       gen_incomplete,                 1_200),
    ("missing_device",           gen_missing_device,             1_500),
    ("clean_response",           gen_clean_response,               800),
    ("social_off_topic",         gen_social_off_topic,             800),

    # ── Action Log & Pronoun Logic ─────────────────────────────────────
    ("action_log_lights",        gen_action_log_lights,          2_000),
    ("action_log_doors",         gen_action_log_doors,           1_500),
    ("action_log_scenes",        gen_action_log_scenes_therm,    1_500),
    ("action_log_gadgets",       gen_action_log_gadgets,          1_000),
    ("undo_multi_action",        gen_undo_multi_action,          1_200),
    ("pronoun_crossroom",        gen_pronoun_crossroom,            900),
    ("room_priority_over_log",   gen_room_priority_over_log,     2_500),
    ("high_count_undo_stress",   gen_high_count_undo_stress,     1_000),
    ("this_vs_that_door",        gen_this_vs_that_door,          1_000),
    ("pronoun_them_door_scope",  gen_pronoun_them_door_scope,    1_500),
    ("disambiguate_back",        gen_back_synonym_disambiguation, 1_200),

    # ── Bulk & Relative State Logic ────────────────────────────────────
    ("bulk_state_aware",         gen_bulk_state_aware,           4_000),
    ("bulk_already_satisfied",   gen_bulk_already_satisfied,     2_500),
    ("relative_clause",          gen_relative_state_clauses,     2_000),
    ("log_plus_rel_clause",      gen_log_plus_relative_clause,     600),
    ("open_unlock_reinforced",   gen_open_unlock_direction_reinforced, 1_500),
    ("full_topology_bulk_doors", gen_full_topology_bulk_doors,   1_000),
    ("lock_all_doors_reinforced",gen_lock_all_doors_reinforced,  1_000),
    ("all_devices_with_room_set",gen_all_devices_with_room_set,  1_500),
    ("bulk_plus_gadget",         gen_bulk_plus_gadget,           1_500),
    ("double_bulk_stress",       gen_double_bulk,                1_500),

    # ── Advanced Compounds ─────────────────────────────────────────────
    ("mixed_compound",           gen_mixed_compound,             2_000),
    ("multi_room_lights",        gen_multi_room_lights,          1_500),
    ("multi_room_doors",         gen_multi_room_doors,           1_500),
    ("compound_scene_device",    gen_compound_scene_device,      3_500),
    ("compound_three_action",    gen_compound_three_action,      2_000),
    ("compound_local",           gen_compound_local,             1_500),
    ("compound_log_undo_plus_action", gen_compound_log_undo_plus_action, 1_500),
    ("compound_three_action_local",   gen_compound_three_action_local,   1_000),
    ("compound_thermostat_incremental", gen_compound_thermostat_incremental, 2_000),
    ("compound_local_door_plus_media", gen_compound_local_door_plus_media, 1_500),
    ("implicit_light_user_room_strict", gen_implicit_light_user_room_strict, 1_500),
    ("compound_relative_plus_gadget", gen_compound_relative_plus_gadget, 1_000),
    ("compound_multi_gadget_explicit", gen_compound_multi_gadget_explicit, 1_500),
    ("partial_execution_ambiguous_local", gen_partial_execution_ambiguous_local, 2_000),
    ("compound_direction_stress", gen_compound_direction_stress, 1_000),
    ("self_contradictory",       gen_self_contradictory_compound, 2_000),

    # ── Gadgets, Media & Math ──────────────────────────────────────────
    ("tv_commands",              gen_tv_commands,                1_000),
    ("speaker_commands",         gen_speaker_commands,           4_500),
    ("fan_commands",             gen_fan_commands,               1_000),
    ("local_media_commands",     gen_local_media_commands,       6_000),
    ("compound_media_plus_devices", gen_compound_media_plus_devices, 6_500),
    ("thermostat_incremental",   gen_thermostat_incremental,     1_500),
    ("rule3_inference",          gen_rule3_inference,            8_000),
    ("speaker_explicit_stop",    gen_speaker_explicit_stop,        800),
    ("speaker_resume_synonyms",  gen_speaker_resume_synonyms,      600),
    ("thermostat_reinforced",    gen_thermostat_reinforced,      1_200),
    ("speaker_on_stop_synonyms", gen_speaker_on_stop_synonyms,   3_000),
    ("speaker_explicit_room_multi", gen_speaker_explicit_room_multi, 3_000),

    # ── Targeted Gap Fixes ─────────────────────────────────────────────
    ("state_grounding",          gen_state_grounding_stress,     3_000),
    ("compound_count",           gen_compound_count_enforcement, 2_000),
    ("them_plurality",           gen_them_plurality,             2_000),
    ("bulk_plus_local_door",     gen_bulk_plus_local_door,       1_500),
    ("door_incomplete_vs_unsupported", gen_door_incomplete_vs_unsupported, 800),
    ("mixed_compound_hallway_boost", gen_mixed_compound_hallway_boost, 800),
    ("action_required_no_ambiguity", gen_action_required_no_ambiguity, 2_000),
    ("state_report_queries",     gen_state_report_queries,       4_500),
    ("living_room_door_positive", gen_living_room_door_positive,   800),
    ("action_log_queries",       gen_action_log_queries,         1_500),
    ("off_everything",           gen_off_everything,             2_000),
    ("pronoun_correct_tool_type", gen_pronoun_correct_tool_type, 1_500),
    ("mixed_device_undo",        gen_mixed_device_block_pronoun_undo, 2_500),
    ("scope_isolation_stress",   gen_scope_isolation_stress,     3_000),
    ("response_text_grounding",  gen_response_text_grounding,    2_000),
    ("direct_command_ignore_log", gen_direct_command_ignore_log, 2_000),
    ("gadget_explicit_room_rejection", gen_gadget_explicit_room_rejection, 1_500),
    ("current_room_unsupported", gen_current_room_unsupported_device, 700),
    ("explicit_room_over_log",   gen_explicit_room_over_log,     2_000),
    ("list_and_local",           gen_list_and_local_device,      1_500),
    ("heterogeneous_undo",       gen_heterogeneous_undo,         1_000),
    ("compound_log_and_local",   gen_compound_log_and_local,     1_500),

    # ── Exhaustive Logic Baseline ──────────────────────────────────────
    ("exhaustive_light_logic",   gen_exhaustive_light_logic,       None),
    ("exhaustive_pronoun_states",gen_exhaustive_pronoun_states,    None),
    ("exhaustive_gadget_rules",  gen_exhaustive_gadget_rules,      None),
    ("exhaustive_compound_pairs",gen_exhaustive_compound_pairs,    None),
]
def proportional_sample(dataset: list, target: int) -> list:
    by_cat = defaultdict(list)
    for ex in dataset: by_cat[ex["category"]].append(ex)
    total_raw = len(dataset)
    result, remainders = [], []
    for cat, exs in by_cat.items():
        exact  = len(exs) / total_raw * target
        floor_ = math.floor(exact)
        result.extend(random.sample(exs, min(floor_, len(exs))))
        remainders.append((exact - floor_, cat))
    slots_left = target - len(result)
    remainders.sort(reverse=True)
    for i in range(min(slots_left, len(remainders))):
        cat   = remainders[i][1]
        exs   = by_cat[cat]
        taken = len([e for e in result if e["category"] == cat])
        if taken < len(exs): result.append(exs[taken])
    random.shuffle(result)
    return result[:target]


def main():
    print("Generating Stage 2 dataset v12 (All Fixes + Positive Think Traces)…\n")
    dataset = []

    for label, fn, tgt in CATEGORY_PLAN:
        if tgt is None:
            batch = fn()
        else:
            batch = fn(tgt)
        print(f"  ✓  {label:<36} {len(batch):>6,}")
        dataset.extend(batch)

    washroom = gen_washroom_boost()
    print(f"  ✓  {'washroom_boost':<36} {len(washroom):>6,}")
    dataset.extend(washroom)

    raw_total = len(dataset)
    print(f"\n  Raw total: {raw_total:,}")
    random.shuffle(dataset)
    dataset = proportional_sample(dataset, FINAL_TARGET)
    total   = len(dataset)
    print(f"  After proportional sampling: {total:,}")

    out_path = "stage2_dataset_v12.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"\n  Written → {out_path}")

    # ── Tool distribution ───────────────────────────────────────────────
    tc_counts = Counter()
    for ex in dataset:
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and "<|tool_call_start|>" in msg["content"]:
                matches = re.findall(r'"name":\s*"([^"]+)"', msg["content"])
                for m in matches: tc_counts[m] += 1

    print("\n  Tool distribution:")
    for name, cnt in sorted(tc_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:<32} {cnt:>6,}  ({cnt/total*100:.1f}%)")

    # ── Coverage stats ──────────────────────────────────────────────────
    think_n = sum(1 for ex in dataset
                  if any("<think>" in (m.get("content") or "")
                         for m in ex["messages"]))
    print(f"\n  <think> trace coverage:          {think_n/total*100:.1f}%")

    # FIX-D verification: count ACTION REQUIRED markers
    action_req_n = sum(1 for ex in dataset
                       for msg in ex["messages"]
                       if msg["role"] == "assistant"
                       and "ACTION REQUIRED." in (msg.get("content") or ""))
    no_action_n  = sum(1 for ex in dataset
                       for msg in ex["messages"]
                       if msg["role"] == "assistant"
                       and "ACTION NOT REQUIRED." in (msg.get("content") or ""))
    print(f"  ACTION REQUIRED markers:         {action_req_n:,}")
    print(f"  ACTION NOT REQUIRED markers:     {no_action_n:,}")

    txn_n = sum(1 for ex in dataset
                if any("[RECENT ACTIONS:" in (m.get("content") or "")
                       for m in ex["messages"]))
    print(f"  Transaction log coverage:        {txn_n/total*100:.1f}%")

    multi_txn = 0
    for ex in dataset:
        for msg in ex["messages"]:
            c = msg.get("content") or ""
            if "[RECENT ACTIONS:" in c:
                blocks = re.findall(r'\[([^\]]+)\]', c)
                if any(',' in b and ('toggle_lights' in b or 'lock_door' in b)
                       for b in blocks):
                    multi_txn += 1
                    break
    print(f"  Multi-call transaction examples: {multi_txn:,} ({multi_txn/total*100:.1f}%)")

    exh_n = sum(1 for ex in dataset if "exhaustive" in ex.get("category", ""))
    print(f"  Exhaustive logic examples:       {exh_n:,} ({exh_n/total*100:.1f}%)")

    # ── Category breakdown ──────────────────────────────────────────────
    cat_counts = Counter(ex["category"] for ex in dataset)
    print("\n  Category breakdown:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        flag = ""
        if "exhaustive" in cat:  flag = " ← exhaustive"
        if cat in ("disambiguate_back", "relative_clause"): flag = " ← v12 new"
        print(f"    {cat:<36} {cnt:>6,}{flag}")

    print("\n  Done.")


if __name__ == "__main__":
    main()