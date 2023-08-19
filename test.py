import re

LINK_PATTERN = r"\b(?:https?://)?(?:(?i:[a-z]+\.)+)[^\s,]+\b"
LINKS = [
    "images-ext-1.discordapp.net",
    "images-ext-2.discordapp.net",
    "images-ext-3.discordapp.net",
    "images-ext-4.discordapp.net",
    "cdn.discordapp.com",
    "tenor.com",
    "giphy.com",
    "media.tenor.com",
    "media.giphy.com",
    "i.imgur.com"
]