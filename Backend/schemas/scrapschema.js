const mongoose=require('mongoose')
const scrapSchema= new mongoose.Schema({
    name:{
        type: String,
        minlength:[1,"Minimum length should be 5"]
    },
    phone:{
        type:String
    },
   
    typeofscrap:{
        type:String

    },
    state:{
        type:String
    },
    district:{
        type:String
    },
    municipality:{
        type:String
    },
    quantity:{
        type:Number
    }
   
},{timestamps:true})
const scrap= mongoose.model("scrap",scrapSchema)
module.exports=scrap