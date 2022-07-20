alter view Project2
as

SELECT distinct
	b.PlayerID,  
	a.nameFirst, 
	a.nameLast, 
	a.nameFirst +' '+ a.nameLast as FullName, 
	cast(a.debut as date) FirstGame, 
	cast(a.finalGame as date) FinalGame, 
	a.bats, 
	a.throws, 
	a.birthCity, 
	a.birthCountry,
	a.birthyear, 
	a.deathYear,
	b.*,
	DATEDIFF(dd, a.birthyear,  isnull(deathyear,year(getdate()))) Age, 
	DATEDIFF(dd,a.debut,a.finalGame) HowLongPlayedDays 
	
FROM [dbo].[People] as A
inner join [dbo].[Batting] as b on b.playerID = a.playerID
where year(finalGame) = 2018

