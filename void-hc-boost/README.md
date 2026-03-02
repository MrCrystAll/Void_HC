[Back to main](../README.md)

# Boost routines

## Boost usage routine

| Python version  | Pylint score |
|---|---|
| 3.10    	| ![Pylint score](badges/pylint_boost_3.10.svg)
| 3.11    	| ![Pylint score](badges/pylint_boost_3.11.svg)
| 3.12    	| ![Pylint score](badges/pylint_boost_3.12.svg)


### State machine

![Boost](../images/Void%20HC-Boost%20usage%20routine.png)

#### Possible scenarios

This state machine is a bit more "complicated", since i have conditions based on the boost amount

On reset, i check whether the boost has boost or not, this defines the starting state

- If no boost -> Empty boost
- If boost -> Is player boosting ?
	- If player boosting -> Boosting
	- If not boosting -> Not boosting

If the player has an empty boost tank, triggering the boost action will not update to Boosting, meaning the boost bin will not be activated.

See [Changelog](CHANGELOG.md)

[Back to top](#boost-routines)