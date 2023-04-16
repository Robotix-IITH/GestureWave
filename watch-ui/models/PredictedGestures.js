const mongoose = require('mongoose');

const PredictedGesturesSchema = new mongoose.Schema({
  _itemId: Number,
  motionName: String,
});

const PredictedGestures =
  mongoose.models.predictedDataV1 ||
  mongoose.model('predictedDataV1', PredictedGesturesSchema);

module.exports = PredictedGestures;
