from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
from datetime import datetime


class ProductBase(BaseModel):
    name: str
    brand: str = "Sin marca"
    category: str = "Sin categoría"
    product_url: str
    source_url: str

    discount_percent: str = "0%"
    rating: str = "Sin calificación"
    original_price: str = "Sin descuento"
    original_price_num: float = 0
    discount_price: str = "0"
    discount_price_num: float = 0

    image_url: Optional[str] = None
    specifications: Dict[str, str] = Field(default_factory=dict)
    availability: str = "Disponible"
    in_stock: bool = True

    source: str = "alkosto"
    scraping_date: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class ProductCreate(ProductBase):
    pass


class ProductResponse(ProductBase):
    id: str = Field(alias="_id")

    @validator('id', pre=True)
    def convert_objectid_to_str(cls, v):
        if v is not None:
            return str(v)
        return v

    class Config:
        allow_population_by_field_name = True


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    discount_percent: Optional[str] = None
    original_price_num: Optional[float] = None
    discount_price_num: Optional[float] = None
    availability: Optional[str] = None
    in_stock: Optional[bool] = None
    last_updated: datetime = Field(default_factory=datetime.now)