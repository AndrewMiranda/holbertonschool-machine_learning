// updateds school that have name Holberton school with address
db.school.updateMany(
    {name: "Holberton school"},
    {"$set": {address: "972 Mission street"}}
);