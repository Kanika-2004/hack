// const { createcart } = require('../repositories/cartRepository');
const {find,create}=require('../repositories/userRepository')
async function registeruser(userdetails){
 
        const user=await find({
            email:userdetails.email,
            mobilenumber: userdetails.mobilenumber
            // password:userdetails.password
        });
        if(user){
            throw{message:'An user already exists',statusCode:400}
        }
        const newuser=await create(userdetails);
        console.log("user service",newuser)
        if(!newuser){
            throw{message:"something went wrong cannot create user"}
        }
       
        // await createcart(newuser?._id)
        return newuser
    
}
module.exports=registeruser