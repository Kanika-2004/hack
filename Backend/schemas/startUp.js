const mongoose = require("mongoose");

const StartupSchema = new mongoose.Schema({
  date: { type: String, required: true },
  startupName: { type: String, required: true },
  industry: { type: String, required: false },
  subVertical: { type: String, required: false },
  city: { type: String, required: false },
  investors: { type: String, required: true },
  investmentType: { type: String, required: true },
  amount: { type: String, required: true },
  remarks: { type: String, required: false },
});

module.exports = mongoose.model("Startup", StartupSchema);
