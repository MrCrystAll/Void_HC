[Back to main](../README.md)

# Flip / dodge routine

|   |   |
|---|---|
| 3.10    	| ![Pylint score](badges/pylint_flip_3.10.svg)
| 3.11    	| ![Pylint score](badges/pylint_flip_3.11.svg)
| 3.12    	| ![Pylint score](badges/pylint_flip_3.12.svg)

## State machine

![Flip](../images/Void%20HC-Flip%20routine.png)

Every transition to "Is flipping" requires the yaw/pitch input from the net.

### Possible scenarios
- On ground -> Is jumping -> Is flipping -> Has flipped -> On ground (Manual flipping)
- On ground -> Is jumping -> Is double jumping -> On ground (Manual double jump)
- On ground -> Is flipping -> Has flipped -> On ground (Flip routine)
- On ground -> Is jumping -> On ground -> (Simple jump)


### When should the bot use this and how ?

- The bot is either targetting the ball or going away from it
- This *theoretically* means that the bot does not give directions, but the ATBA PID does.
- Should i allow the bot to override these values ?
	- If yes, what action is responsible for that ? A potential direction perturbator ? Adding a delta to the current target (going to right/left or up/down of the current target) -> would this be additive ? should i add an input to the bot that could give a -1/1 on the yaw/pitch axis ? Why not just do an action that when used considers the yaw/pitch input as a target displacement input ? Is this precise enough, it sounds like manually aiming with a tank on a supersonic target...TBD
	- If no, what are the limitations ? Potentially messing up a shooting behavior ? Will the bot even be able to score if the ball - player alignement does not face the net at all ? Previous runs have shown that the bot doesn't use the go away from ball action to modify its direction (maybe a reward issue ?)

### How do i set this up ?

Ideally i'd like to first create a routine that flips in a direction, no ATBA input, just flip towards a given direction.

Once i did that, i can try to integrate it with the ATBA behavior, since the flip routine doesn't overwrite the yaw/pitch input, this should allow the base ATBA to be faster (although i will not make it auto flip using only ATBA inputs, i believe that it should be the decision of the ML agent).

Going more on the ML side, this means that the bot cannot flip elsewhere than towards the ball (or roughly close to the ball, assuming i allow overriding the values) -> Is it good ? I would say yes, at first, might become detrimental as time goes on, but that's not my problem, that's future me's problem.

Following the state machine's transitions, only one action is necessary to trigger the routine, and this action uses a yaw/pitch input, which is either 0 (double jump?) if no atba and no manual overriding, or not 0 (atba / manual). So in both cases, it's fine, although i believe it can't really be 0 since there's always some ATBA.

If i allow the manual flipping, that means the jump input is not tied to the yaw/pitch, but only the flip part of it is.
But what if it "jumps" while already jumping ? Should i consider this action as flipping ? Should i consider a double jump as a "is flipping" state -> technically that's a 0 yaw/pitch flip right?

I finished talking about it here: https://discord.com/channels/348658686962696195/1476251762150346882/1477066491705688124

[Back to top](#flip--dodge-routine)