# Lua / Void HC

This bot is an experimentation i'm going to try when i want to, it is an hybrid bot whose only objective is to learn actual game sense and mechanics rather than just driving to the ball.

The whole point of the experiment is to use hardcoded routines and knowledge usually made for hardcoded bots, and use that within a ML agent to see whether it can use those advances routines to play better.

I'm using the name "Lua" but realistically it's just an HC experiment for Void, hence the "/".

## Routines used in this bot

| Routine 	| Quick summary                                  	| Documentation                          	|
|---------	|------------------------------------------------	|----------------------------------------	|
| ATBA    	| A routine to go towards / away from the ball   	| [Documentation](void-hc-atba/README.md) 	|
| Flip    	| A routine to make the bot flip towards a point 	| [Documentation](void-hc-flip/README.md) 	|

## Motivation for this project

I've been training Rocket League ML agents for almost 3 years now, and i believe in the fact that more computing power will not always yield better results. I believe in creativity and optimization.

Most of the early training in ML within Rocket League is always the same. Train the bot to go towards the ball, make it discover how to touch the ball, how to shoot it, how to not get scored on, how to score. This is repetitive, this is time consuming, this is boring.