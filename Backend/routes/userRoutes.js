const createuser=require('../controllers/userControllers')
const express=require('express')
const userrouter=express.Router()
userrouter.post('/',createuser)
module.exports=userrouter