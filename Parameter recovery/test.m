clear all;
dataSource = sprintf('gamble_%d_all_sessions_permuted=%s',whichGamble,permuted);%All data needed to simulate choices for specific gamble
load(dataSource)

t