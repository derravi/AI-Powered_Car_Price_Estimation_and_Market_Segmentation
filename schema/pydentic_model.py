from pydantic import BaseModel, Field
from typing import Annotated

class UserInput(BaseModel):
    brand: Annotated[str, Field(..., description="Enter the Car Brand (e.g., Maruti Wagon)")]
    unique_model_number: Annotated[str, Field(..., description="Enter the unique model number (e.g., R LXI Minor)")]
    year: Annotated[int, Field(..., gt=0, description="Enter the Manufacturing Year (e.g., 2019)")]
    fuel: Annotated[str, Field(..., description="Enter Fuel Type (Petrol/Diesel/CNG/LPG/Electric)")]
    transmission: Annotated[str, Field(..., description="Enter Transmission Type (Manual/Automatic)")]
    km_driven: Annotated[float, Field(..., gt=0, description="Enter Total KM Driven (e.g., 12000.5)")]
    owner: Annotated[str, Field(..., description="Enter Owner Type (First Owner/Second Owner/Third Owner/etc.)")]
    seller_type: Annotated[str, Field(..., description="Enter Seller Type (Dealer/Individual/Trustmark Dealer)")]
