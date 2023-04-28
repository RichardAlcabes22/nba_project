# nba_classification_project
#### Welcome to this initial exploration of NBA Team Data from the 1979-80 season until the current 2022-23 season!  The primary purpose of this project is to make accurate mid-season predictions about which teams will make the playoffs at the end of the season. The project data-set was personally curated coutesy of https://www.basketball-reference.com/.  Season-wide Team data for each NBA Team from 1979 until present was aggregated for each team and each row was labeled as a "Playoff" team or a "Non-Playoff" team.
#### The goals of this initial exploration are as follows:
- PRIMARY: Generate actionable predictions regarding whether or not a team will qualify for the Playoffs at season's-end.
- SECONDARY: Draw reasonable inferences regarding the characteristics of those teams which qualify for the playoffs, and those that do not.
- SECONDARY: Identify those features which retain the greatest importance for determining the playoff status of a particular team.

#### PROJECT DESCRIPTION and ASSUMPTIONS:
- Basketball has changed significantly over the past several generations, with respect to atctics, strategy, and tempo.  Thus, the team statistics generated per game have changed over time to reflect this assertion. 
- Therefore, this dataset does not display team stats on a "per game" basis.  Instead, team stats are displayed "per 100 possessions".  This allows for a somewhat-valid comparison to be made between teams of differing generations.  This also partially controls for stats being functional dependent upon common in-game conditions and provides a window into a more distilled view of a team's capabilities.
- The 1979-80 NBA Season was chosen as the starting point for the dataset because it represesnts the most recent, fundamental change to the game: the 3-pt Shot.  While, it took over 20 years for the 3-pt shot to actually change the game, it is important to note that the efficacy of the model is assumed to be correlated with the number of datapoints (approx. 1200), and thus, 1979 was chosen over 2003, as this would only yield aprox. 600 datapoints.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from from https://www.basketball-reference.com/ by subsetting team-level statistics which include stats which reflect "skill, talent, and hustle".  For example, team Field Goal % is a function of skill level(among other contributors), while Steals, Rebounds, or Fouls Drawn are a function of "talent and hustle" (among others).
- Prepare: Kept outliers after investigating their nature, missingness was a non-issue, as there were ZERO entries containing NULL values for predictors.  StandardScaler used for scaling purposed.  Split into ML subsets (Train/Validate/Test).
- Explore: Univariate and multi-variate analysis, correlation matrix, 2D visualization, correlation significance testing, 2-sample T-testing.
- Model: Established a baseline "Precision" for Positive class of 56.5% using the most frequent target occurance of "yes: playoffs".  Then with a Vanilla, Obviously Overfit, DecisionTreeClassifier with default hyperparameters, established a new Precision floor of 82.0%. After trying different tree-based and non-tree-based algorithms with differing tunings, findings indicated a DecisionTree Classifier with a max_depth setting of 4 yielded best results (88.0% Precision on Test).
- Deliver: Please refer to this doc as well as the Final_NBA.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.