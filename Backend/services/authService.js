const {find}=require('../repositories/userRepository')
const jwt=require('jsonwebtoken')
const bcrypt=require('bcrypt')
const { JWT_SECRET, JWT_EXPIRY } = require('../config/serverConfig')
async function login(userdetails){
    const email=userdetails.email
    const plainpassword=userdetails.password
    const user=await find({email})
    if(!user){
        throw{message:"no user found"}
        
    }
    const response= await bcrypt.compare(plainpassword,user.password)
    if(!response){
        throw{message:"wrong password"}
        
    }
    const token=jwt.sign({email:user.email,id:user._id},JWT_SECRET,{
        expiresIn:JWT_EXPIRY
    })
    console.log("The token is",token)
    return token
}
module.exports=login