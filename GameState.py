from enum import Enum


class Person:
    def __init__(self):
        self.Virtue = Trait.Unknown;
        self.Vice = Trait.Unknown;


class Trait(Enum):
    Unknown = 0,
    Suspicion = (1 << 0),  # Check this bit if its an unknown tier
    Tier1 = (1 << 1),
    Tier2 = (1 << 2),
    Tier3 = (1 << 3),
    Tier4 = (1 << 4),
    Tier5 = (1 << 5),
    Tier6 = (1 << 6),
    Major = Tier6 | Tier5,  # We can or to see if these bits are set, only teir 5 and 6 have these bits set, they are the major vices/virtues
    Virtue = (1 << 7),  # Check for vice if this bit is missing
    Ignorance = (1 << 8),
    Fervor = (1 << 9),
    Discipline = (1 << 10),
    Penitence = (1 << 11),
    Obedience = (1 << 12)

class NamedTrait(Enum):
    Childlike = Trait.Tier1 | Trait.Virtue | Trait.Ignorance,
    Practical = Trait.Tier2 | Trait.Virtue | Trait.Ignorance,
    Dull = Trait.Tier3 | Trait.Virtue | Trait.Ignorance,
    Uneducated = Trait.Tier4 | Trait.Virtue | Trait.Ignorance,
    Superstitious = Trait.Tier6 | Trait.Virtue | Trait.Ignorance,

    Charismatic = Trait.Tier1 | Trait.Virtue | Trait.Fervor,
    Violent = Trait.Tier3 | Trait.Virtue | Trait.Fervor,
    Singer = Trait.Tier4 | Trait.Virtue | Trait.Fervor,
    Sadist = Trait.Tier5 | Trait.Virtue | Trait.Fervor,
    Pyromaniac = Trait.Tier6 | Trait.Virtue | Trait.Fervor,