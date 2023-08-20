# Scouting Classification with Machine Learning

![Project Image](https://www.sportsbusinessjournal.com/-/media/Sporttechie/2022/08/17/soccer_scouting_part_3_lead_image.ashx?mw=768)) <!-

## Project Description

This project includes salary prediction algorithms used with machine learning, aiming to predict the class (average, highlighted) players belong to based on the scores given to the characteristics of football players observed by scouts.

## Business Problem

The goal is to predict the class of players (average or highlighted) based on the scores assigned by scouts to specific player attributes.

## Data Set Story

The dataset originates from Scoutium and comprises information about the attributes and scores given by scouts to football players based on their observed characteristics during matches.

### `scoutium_attributes.csv`

- `task_response_id`: A set of evaluations by a scout for all players on a team's roster in a match
- `match_id`: The identifier of the relevant match
- `evaluator_id`: The identifier of the evaluator (scout)
- `player_id`: The identifier of the relevant player
- `position_id`: The identifier of the position played by the player in that match
- `analysis_id`: A set of attribute evaluations by a scout for a player in a match
- `attribute_id`: The identifier of each attribute for which players were evaluated
- `attribute_value`: The value (points) assigned by a scout to a player's attribute

### `scoutium_potential_labels.csv`

- `task_response_id`: A set of evaluations by a scout for all players on a team's roster in a match
- `match_id`: The identifier of the relevant match
- `evaluator_id`: The identifier of the evaluator (scout)
- `player_id`: The identifier of the relevant player
- `potential_label`: A label indicating the final decision of a scout regarding a player in a match (target variable)

