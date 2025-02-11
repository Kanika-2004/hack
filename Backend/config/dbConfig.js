const mongoose=require('mongoose')
const serverconfig=require('./serverConfig.js')
async function connectDb(){
    try{
        await mongoose.connect(serverconfig.DB_URL)
        console.log("successfully connected to database");
    }
    catch(error){
        console.log("unable to connect to database")
        console.log(error)
    }
}
module.exports=connectDb